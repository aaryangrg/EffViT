# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchpack.distributed as dist
from tqdm import tqdm

from efficientvit.apps.trainer import Trainer
from efficientvit.apps.utils import AverageMeter, sync_tensor
from efficientvit.clscore.trainer.utils import accuracy, apply_mixup, label_smooth
from efficientvit.models.utils import list_join, list_mean, torch_random_choices

__all__ = ["ClsMutualTrainer"]
LOG_SOFTMAX_CONST = 1e-6
PREDEFINED_WIDTHS = [0.25, 0.50, 0.75, 1.0]

class ClsMutualTrainer(Trainer):
    def __init__(
        self,
        path: str,
        model: nn.Module,
        p_model : nn.Module,
        data_provider,
        auto_restart_thresh: float or None = None,
    ) -> None:
        super().__init__(
            path=path,
            model=model,
            data_provider=data_provider,
        )
        self.auto_restart_thresh = auto_restart_thresh
        self.test_criterion = nn.CrossEntropyLoss()
        self.p_model = p_model
        self.p_model.cuda()
        # Freeze the parameters of the pre-trained model
        for param in self.p_model.parameters():
            param.requires_grad = False
        self.p_model.eval()
        self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))

    def _validate(self, model, data_loader, epoch) -> dict[str, any]:

        val_loss = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()
        
        #Validate on max-width
        self.model.apply(lambda m: setattr(m, 'width_mult', PREDEFINED_WIDTHS[-1]))
        
        with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc=f"Validate Epoch #{epoch + 1}",
                disable=not dist.is_master(),
                file=sys.stdout,
            ) as t:
                for images, labels in data_loader:
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = model(images)
                    loss = self.test_criterion(output, labels)
                    val_loss.update(loss, images.shape[0])
                    if self.data_provider.n_classes >= 100:
                        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                        val_top5.update(acc5[0], images.shape[0])
                    else:
                        acc1 = accuracy(output, labels, topk=(1,))[0]
                    val_top1.update(acc1[0], images.shape[0])

                    t.set_postfix(
                        {
                            "loss": val_loss.avg,
                            "top1": val_top1.avg,
                            "top5": val_top5.avg,
                            "#samples": val_top1.get_count(),
                            "bs": images.shape[0],
                            "res": images.shape[2],
                        }
                    )
                    t.update()
        return {
            "val_top1": val_top1.avg,
            "val_loss": val_loss.avg,
            **({"val_top5": val_top5.avg} if val_top5.count > 0 else {}),
        }

    def before_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        images = feed_dict["data"].cuda()
        labels = feed_dict["label"].cuda()

        # label smooth
        labels = label_smooth(labels, self.data_provider.n_classes, self.run_config.label_smooth)

        # mixup
        if self.run_config.mixup_config is not None:
            # choose active mixup config
            mix_weight_list = [mix_list[2] for mix_list in self.run_config.mixup_config["op"]]
            active_id = torch_random_choices(
                list(range(len(self.run_config.mixup_config["op"]))),
                weight_list=mix_weight_list,
            )
            active_id = int(sync_tensor(active_id, reduce="root"))
            active_mixup_config = self.run_config.mixup_config["op"][active_id]
            mixup_type, mixup_alpha = active_mixup_config[:2]

            lam = float(torch.distributions.beta.Beta(mixup_alpha, mixup_alpha).sample())
            lam = float(np.clip(lam, 0, 1))
            lam = float(sync_tensor(lam, reduce="root"))

            images, labels = apply_mixup(images, labels, lam, mixup_type)

        return {
            "data": images,
            "label": labels,
        }

    # Using pre-defined fixed widths (0.25, 0.50, 0.75, 1.0) [MutualNet scheme with fixed widths]
    def run_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        images = feed_dict["data"]
        labels = feed_dict["label"]
        
        # Mesa --> null by default
        # if self.run_config.mesa is not None and self.run_config.mesa["thresh"] <= self.run_config.progress:
        #     ema_model = self.ema.shadows
        #     with torch.inference_mode():
        #         ema_output = ema_model(images).detach()
        #     ema_output = torch.clone(ema_output)
        #     ema_output = F.sigmoid(ema_output).detach()
        # else:
        #     ema_output = None

        with torch.autograd.set_detect_anomaly(True) :
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
                # p_output = self.p_model(images)

                # Max-width
                # self.model.apply(lambda m: setattr(m, 'width_mult', PREDEFINED_WIDTHS[-1]))
                max_width_output = self.model(images)

                #Task Loss - Max-width
                loss = self.train_criterion(max_width_output, labels)
                ce_loss = loss
                total_kd_loss = 0
                max_width_output_detached = max_width_output.detach()
            self.scaler.scale(loss).backward()
            
            # Bears significant computational overhead
            for width_mult in (PREDEFINED_WIDTHS[:len(PREDEFINED_WIDTHS)-1])[::-1]:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
                    with torch.no_grad():
                        self.model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    output = self.model(images)
                    kd_loss = self.get_kld_loss(output + LOG_SOFTMAX_CONST, max_width_output_detached + LOG_SOFTMAX_CONST)
                    total_kd_loss += kd_loss
                self.scaler.scale(kd_loss).backward()

            
                # mesa loss (Not included by default)
                # if ema_output is not None:
                #     mesa_loss = self.train_criterion(output, ema_output) # Calculated only on CrossEntropy loss
                #     loss = loss + total_kd_loss + self.run_config.mesa["ratio"] * mesa_loss
                # else :
                #     pass
            
        # calc train top1 acc
        if self.run_config.mixup_config is None:
            top1 = accuracy(max_width_output_detached, torch.argmax(labels, dim=1), topk=(1,))[0][0]
        else:
            top1 = None

        return {
            "loss": ce_loss + total_kd_loss,
            "top1": top1,
            "task_loss" : ce_loss,
            "kd_loss" : total_kd_loss
        }
    
    def get_kld_loss(self,scale_pred, scale_soft, temperature = 1.0):
        p_s = F.log_softmax(scale_pred / temperature, dim=1)
        p_t = F.softmax(scale_soft / temperature, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='mean')
        return loss

    def _train_one_epoch(self, epoch: int) -> dict[str, any]:
        train_loss = AverageMeter()
        train_top1 = AverageMeter()
        task_loss = AverageMeter()
        kd_loss = AverageMeter()

        with tqdm(
            total=len(self.data_provider.train),
            desc="Train Epoch #{}".format(epoch + 1),
            disable=not dist.is_master(),
            file=sys.stdout,
        ) as t:
            for images, labels in self.data_provider.train:
                feed_dict = {"data": images, "label": labels}

                # preprocessing
                feed_dict = self.before_step(feed_dict)
                # clear gradient
                self.optimizer.zero_grad()
                # forward & backward
                output_dict = self.run_step(feed_dict)
                # update: optimizer, lr_scheduler
                self.after_step()

                # update train metrics
                train_loss.update(output_dict["loss"], images.shape[0])
                if output_dict["top1"] is not None:
                    train_top1.update(output_dict["top1"], images.shape[0])
                task_loss.update(output_dict["task_loss"], images.shape[0])
                kd_loss.update(output_dict["kd_loss"], images.shape[0])

                # tqdm
                postfix_dict = {
                    "loss": train_loss.avg,
                    "top1": train_top1.avg,
                    "task_loss" : task_loss.avg,
                    "kd_loss" : kd_loss.avg,
                    "bs": images.shape[0],
                    "res": images.shape[2],
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
            **({"train_top1": train_top1.avg} if train_top1.count > 0 else {}),
            "train_loss": train_loss.avg,
            "task_loss" : task_loss.avg,
            "kd_loss" : kd_loss.avg
        }

    def train(self, trials=0, save_freq=1) -> None:
        if self.run_config.bce:
            self.train_criterion = nn.BCEWithLogitsLoss()
        else:
            self.train_criterion = nn.CrossEntropyLoss()

        for epoch in range(self.start_epoch, self.run_config.n_epochs + self.run_config.warmup_epochs):
            train_info_dict = self.train_one_epoch(epoch)
            # eval
            val_info_dict = self.multires_validate(epoch=epoch)
            avg_top1 = list_mean([info_dict["val_top1"] for info_dict in val_info_dict.values()])
            is_best = avg_top1 > self.best_val
            self.best_val = max(avg_top1, self.best_val)

            if self.auto_restart_thresh is not None:
                if self.best_val - avg_top1 > self.auto_restart_thresh:
                    self.write_log(f"Abnormal accuracy drop: {self.best_val} -> {avg_top1}")
                    self.load_model(os.path.join(self.checkpoint_path, "model_best.pt"))
                    return self.train(trials + 1, save_freq)

            # log
            val_log = self.run_config.epoch_format(epoch)
            val_log += f"\tval_top1={avg_top1:.2f}({self.best_val:.2f})"
            val_log += "\tVal("
            for key in list(val_info_dict.values())[0]:
                if key == "val_top1":
                    continue
                val_log += f"{key}={list_mean([info_dict[key] for info_dict in val_info_dict.values()]):.2f},"
            val_log += ")\tTrain("
            for key, val in train_info_dict.items():
                val_log += f"{key}={val:.2E},"
            val_log += (
                f'lr={list_join(sorted(set([group["lr"] for group in self.optimizer.param_groups])), "#", "%.1E")})'
            )
            self.write_log(val_log, prefix="valid", print_log=False)

            # save model
            if (epoch + 1) % save_freq == 0 or (is_best and self.run_config.progress > 0.8):
                self.save_model(
                    only_state_dict=False,
                    epoch=epoch,
                    model_name="model_best.pt" if is_best else "checkpoint.pt",
                )
