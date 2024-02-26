# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import os
from re import M
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchpack.distributed as dist
from tqdm import tqdm

from efficientvit.apps.trainer import Trainer
from efficientvit.apps.utils import AverageMeter, metric, sync_tensor
from efficientvit.clscore.trainer.utils import accuracy, apply_mixup, label_smooth
from efficientvit.models.utils import list_join, list_mean, torch_random_choices
from efficientvit.apps.data_provider.base import parse_image_size
from efficientvit.apps.data_provider import DataProvider, parse_image_size
from efficientvit.apps.trainer.run_config import RunConfig
from efficientvit.apps.utils import EMA
from efficientvit.models.nn.norm import reset_bn
from efficientvit.models.utils import is_parallel, load_state_dict_from_file

__all__ = ["GdinoBackboneTrainer"]
LOG_SOFTMAX_CONST = 1e-6
PREDEFINED_WIDTHS = [0.25, 0.50, 0.75, 1.0]

class GdinoBackboneTrainer(Trainer):
    def __init__(
        self,
        path: str,
        vit_backbone: nn.Module,
        dino_backbone : nn.Module,
        data_provider,
        auto_restart_thresh: float or None = None,
        metric_logger = None
    ) -> None:
        super().__init__(
            path=path,
            model=vit_backbone,
            data_provider=data_provider,
        )
        self.auto_restart_thresh = auto_restart_thresh
        self.dino_backbone = dino_backbone
        self.dino_backbone.eval()
        self.metric_logger = metric_logger


    def prep_for_training_custom(self, run_config: RunConfig, ema_decay: float or None = None, fp16=False) -> None:
        self.run_config = run_config

        self.run_config.global_step = 0
        self.run_config.batch_per_epoch = len(self.data_provider)
        assert self.run_config.batch_per_epoch > 0, "Training set is empty"

        # build optimizer
        self.optimizer, self.lr_scheduler = self.run_config.build_optimizer(self.model)

        # fp16
        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        print("Optimizer and lr scheduler created !")
        # self.test_criterion = nn.CrossEntropyLoss()
    
    # Val should run the validation loop of OpenGDino with a new backbone / new model architecture

    # def _validate(self, model, data_loader, epoch) -> dict[str, any]:
    #     results = []
    #     for width in PREDEFINED_WIDTHS :

    #         with torch.no_grad() :
    #             model.apply(lambda m: setattr(m, 'width_mult', width))
    #             # Reset bn & set to eval mode
    #             if self.run_config.reset_bn :
    #                 self.reset_bn(
    #                     network=model,
    #                     subset_size=self.run_config.reset_bn_size,
    #                     subset_batch_size=self.run_config.reset_bn_batch_size,
    #                     progress_bar=True,
    #                 )

    #         # All validation performed on val
    #         model.eval()

    #         val_loss = AverageMeter()
    #         val_top1 = AverageMeter()
    #         val_top5 = AverageMeter()
            
    #         with torch.no_grad():
    #             with tqdm(
    #                 total=len(data_loader),
    #                 desc=f"Validate Epoch #{epoch + 1} - {width}x",
    #                 disable=not dist.is_master(),
    #                 file=sys.stdout,
    #             ) as t:
    #                 for images, labels in data_loader:
    #                     images, labels = images.cuda(), labels.cuda()
    #                     # compute output
    #                     output = model(images)
    #                     loss = self.test_criterion(output, labels)
    #                     val_loss.update(loss, images.shape[0])
    #                     if self.data_provider.n_classes >= 100:
    #                         acc1, acc5 = accuracy(output, labels, topk=(1, 5))
    #                         val_top5.update(acc5[0], images.shape[0])
    #                     else:
    #                         acc1 = accuracy(output, labels, topk=(1,))[0]
    #                     val_top1.update(acc1[0], images.shape[0])

    #                     t.set_postfix(
    #                         {
    #                             "loss": val_loss.avg,
    #                             "top1": val_top1.avg,
    #                             "top5": val_top5.avg,
    #                             "#samples": val_top1.get_count(),
    #                             "bs": images.shape[0],
    #                             "res": images.shape[2],
    #                         }
    #                     )
    #                     t.update()
    #         results.append({
    #             f"val_top1_{width}": val_top1.avg,
    #             f"val_loss_{width}": val_loss.avg,
    #             **({f'val_top5_{width}': val_top5.avg} if val_top5.count > 0 else {}),
    #         })
        
    #     return results

    def prestep(self, samples) :
        
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = gdino_misc.nested_tensor_from_tensor_list(samples)

        # Zero grad if any accumulates
        self.optimizer.zero_grad()

        return samples
    
    # # Using pre-defined fixed widths (0.25, 0.50, 0.75, 1.0)
    def runstep(self,  samples) -> dict[str, any]:

        # Put model to train
        self.model.train()
        self.dino_backbone.eval()

        # Use half-precision training
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            samples = samples.to("cuda")
            dino_backbone_outputs = []
            dino_backbone_features, __ = self.dino_backbone.backbone(samples)
            for l, feat in enumerate(dino_backbone_features):
                src, mask = feat.decompose()
                dino_backbone_outputs.append(src)
            with torch.no_grad() :
                self.model.apply(lambda m: setattr(m, 'width_mult', PREDEFINED_WIDTHS[-1]))
            
            max_width_outputs = self.model(samples.tensors) # ViT Backbone outputs - should also include masks (Feature pyramid)
            total_kd_loss = 0
            max_width_kd_loss = self.get_kld_loss(max_width_outputs[1:],dino_backbone_outputs)
            total_kd_loss += max_width_kd_loss
        
        max_width_output_detached = []
        for feature in max_width_outputs :
            max_width_output_detached.append(feature.detach())
        self.scaler.scale(max_width_kd_loss).backward()
        
        # Bears significant computational overhead for training
        for width_mult in (PREDEFINED_WIDTHS[:len(PREDEFINED_WIDTHS)-1])[::-1]:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
                with torch.no_grad():
                    self.model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                vit_outputs = self.model(samples.tensors)
                kd_loss = self.get_kld_loss(vit_outputs[1:], max_width_output_detached[1:])
            self.scaler.scale(kd_loss).backward()
            total_kd_loss += kd_loss
    
        return {
            "loss": total_kd_loss,
        }
    
    def get_kld_loss(self,scale_pred, scale_soft, temperature = 1.0):
        loss = 0
        for _ in range(len(scale_pred)) :
            p_s = F.log_softmax(scale_pred[_] + LOG_SOFTMAX_CONST / temperature, dim=1)
            p_t = F.softmax(scale_soft[_] + LOG_SOFTMAX_CONST / temperature, dim=1)
            loss += F.kl_div(p_s, p_t, reduction='mean')
        return loss

    def _train_one_epoch(self, epoch: int) -> dict[str, any]:
    
        train_loss = AverageMeter(False)

        with tqdm(
            total=len(self.data_provider),
            desc="Train Epoch #{}".format(epoch + 1),
            disable=not dist.is_master(),
            file=sys.stdout,
        ) as t:
            for samples, targets in self.data_provider:
                # preprocessing
                samples = self.prestep(samples)
                # clear gradient
                self.optimizer.zero_grad()
                # forward & backward
                output_dict = self.runstep(samples)
                # update: optimizer, lr_scheduler
                self.after_step()

                # update train metrics
                train_loss.update(output_dict["loss"])

                # tqdm
                postfix_dict = {
                    "loss": train_loss.avg,
                    "lr": list_join(
                        sorted(set([group["lr"] for group in self.optimizer.param_groups])),
                        "#",
                        "%.1E",
                    ),
                    "progress": self.run_config.progress,
                }
                t.set_postfix(postfix_dict)
                t.update()
        return {
            "loss" : train_loss.avg
        }

    def train(self, trials=0, save_freq=1) -> None:

        for epoch in range(self.start_epoch, self.run_config.n_epochs + self.run_config.warmup_epochs):
            train_info_dict = self._train_one_epoch(epoch)
            
            # log
            val_log = self.run_config.epoch_format(epoch)
            val_log += ")\tTrain("
            for key, val in train_info_dict.items():
                val_log += f"{key}={val:.2E},"
            val_log += (
                f'lr={list_join(sorted(set([group["lr"] for group in self.optimizer.param_groups])), "#", "%.1E")})'
            )
            self.write_log(val_log, prefix="valid", print_log=False)

            # save model
            self.save_model(
                only_state_dict=False,
                epoch=epoch,
                model_name="model_best.pt",
            )

    # def validate(self, model=None, data_loader=None, is_test=True, epoch=0, reset_bn = False) -> dict[str, any]:
    #     model = model or self.eval_network
    #     if data_loader is None:
    #         if is_test:
    #             data_loader = self.data_provider.test
    #         else:
    #             data_loader = self.data_provider.valid
    #     model.eval()
    #     return self._validate(model, data_loader, epoch)


    # def multires_validate(
    #     self,
    #     model=None,
    #     data_loader=None,
    #     is_test=True,
    #     epoch=0,
    #     eval_image_size=None,
    # ) -> dict[str, dict[str, any]]:
    #     eval_image_size = eval_image_size or self.run_config.eval_image_size
    #     eval_image_size = eval_image_size or self.data_provider.image_size
    #     model = model or self.eval_network

    #     if not isinstance(eval_image_size, list):
    #         eval_image_size = [eval_image_size]

    #     output_dict = {}
    #     for r in eval_image_size:
    #         self.data_provider.assign_active_image_size(parse_image_size(r))
    #         # Batch Norm reset performed inside validation loop
    #         output_dict[f"r{r}"] = self.validate(model, data_loader, is_test, epoch)
    #     return output_dict