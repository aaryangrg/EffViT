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

__all__ = ["GdinoBackboneTrainerNoFlex"]
LOG_SOFTMAX_CONST = 1e-6
PREDEFINED_WIDTHS = [0.25, 0.50, 0.75, 1.0]

class GdinoBackboneTrainerNoFlex(Trainer):
    def __init__(
        self,
        path: str,
        effvit_dino: nn.Module,
        gdino_backbone : nn.Module,
        data_provider,
        auto_restart_thresh: float or None = None,
        metric_logger = None,
        train_full_flexible_model = True,
        fp16_training = False,
        kd_metric = "kld",
    ) -> None:
        super().__init__(
            path=path,
            model=effvit_dino, # Student model (backbone) wrapped within GroundingDINO model (for eval)
            data_provider=data_provider,
        )
        self.auto_restart_thresh = auto_restart_thresh
        self.gdino_backbone = gdino_backbone # Teacher
        
        self.metric_logger = metric_logger
        self.train_full_flexible_model = train_full_flexible_model
        self.fp16_training = fp16_training

        for param in self.gdino_backbone.parameters():
            param.requires_grad = False
        self.gdino_backbone.eval()
        self.kd_metric = kd_metric
        if self.kd_metric == "kld" :
            self.loss_criterion = self.get_kld_loss
        elif self.kd_metric == "ce" :
            self.loss_criterion = self.custom_ce_loss
        elif self.kd_metric == "l2" :
            self.loss_criterion = self.custom_mse_loss
        else :
            self.loss_criterion = self.custom_rmse_loss

    def prep_for_training_custom(self, run_config: RunConfig, ema_decay: float or None = None, fp16=False) -> None:
        self.run_config = run_config

        self.run_config.global_step = 0
        self.run_config.batch_per_epoch = len(self.data_provider)
        assert self.run_config.batch_per_epoch > 0, "Training set is empty"

        # build optimizer
        self.optimizer, self.lr_scheduler = self.run_config.build_optimizer(self.model.effvit_backbone)

        # fp16
        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

    def prestep(self, samples) :
        # Zero grad if any accumulates
        self.optimizer.zero_grad()
        return samples
    
    def runstep(self,  samples) -> dict[str, any]:
        # Put model to train --> ensure rest of model has frozen params
        self.model.effvit_backbone.train()
        with torch.no_grad() :
            self.model.effvit_backbone.apply(lambda m: setattr(m, 'width_mult', PREDEFINED_WIDTHS[-1]))

        # Use half-precision training
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16_training):
            samples = samples.to("cuda")
            dino_backbone_outputs = []
            # This is Joiner.backbone() --> Swin Transfomer directly without positional embeds
            dino_backbone_features = self.gdino_backbone(samples)
            # dino_backbone_features = {idx : NestedTensor(output, mask),}
            for k in dino_backbone_features :
                src, mask = dino_backbone_features[k].decompose()
                dino_backbone_outputs.append(src) 
            
            max_width_outputs = self.model.effvit_backbone(samples.tensors) # list of features in order (mask + position embeds) generated later
            total_kd_loss = 0
            max_width_kd_loss = self.loss_criterion(max_width_outputs[1:], dino_backbone_outputs)
            total_kd_loss += max_width_kd_loss
            # Backward pass on multi-scale KD-loss (added)
        self.scaler.scale(max_width_kd_loss).backward()

        return {
            "loss": total_kd_loss,
        }
    
    def custom_mse_loss(self, scale_pred, scale_soft):
        loss = 0
        loss_fn = nn.MSELoss()
        for pred, soft in zip(scale_pred, scale_soft):
            loss += loss_fn(pred, soft)
        return loss

    def custom_rmse_loss(self, scale_pred, scale_soft):
        loss = 0
        loss_fn = nn.MSELoss()
        for pred, soft in zip(scale_pred, scale_soft):
            loss += torch.sqrt(loss_fn(pred, soft))
        return loss
    
    def custom_ce_loss(self, scale_pred, scale_soft) :
        loss = 0
        criterion = nn.CrossEntropyLoss()
        for _ in range(len(scale_pred)) :
            loss += criterion(scale_pred[_], scale_soft[_])
        return loss
    
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

    def train(self, trials=0, save_freq=1, criterion = None, postprocessors = None, data_loader_val = None, base_ds = None, args = None, evaluate_custom = None) -> None:
    
        for epoch in range(self.start_epoch, self.run_config.n_epochs + self.run_config.warmup_epochs):

            train_info_dict = self._train_one_epoch(epoch)

             # Validate one epoch (using GDINO validation loop) --> BN reset not required for single resolution
            self.model.effvit_backbone.apply(lambda m: setattr(m, 'width_mult', PREDEFINED_WIDTHS[-1]))
            test_stats, coco_evaluator = evaluate_custom(self.model, criterion, postprocessors,data_loader_val, base_ds, "cuda", wo_class_error=False, args=args)

            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
            
            val_log = self.run_config.epoch_format(epoch)
            for k in test_stats :
                val_log = val_log + f"val_{k} : {test_stats[k]} | "
            val_log += ")\tTrain("
            for key, val in train_info_dict.items():
                val_log += f"{key}={val:.2E},"
            val_log += (
                f'lr={list_join(sorted(set([group["lr"] for group in self.optimizer.param_groups])), "#", "%.1E")})'
            )
            self.write_log(val_log, prefix="valid", print_log=False)

            # Save model if val mAP > current mAP
            self.save_model(
                only_state_dict=False,
                epoch=epoch,
                model_name="model_best.pt",
            )