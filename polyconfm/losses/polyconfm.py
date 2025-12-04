# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from sklearn.metrics import r2_score


@register_loss("polyconfm_pretrain_phase1")
class PolyConFMPretrainPhase1Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        loss = model(**sample["net_input"])
        logging_output = {
            "loss": loss.data,
            "sample_size": 1,
            "bsz": 1
        }
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
       
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return is_train
       

@register_loss("polyconfm_pretrain_phase2")
class PolyConFMPretrainPhase2Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        loss, rot_loss, dist_mat_loss = model(**sample["net_input"])
        logging_output = {
            "loss": loss.data,
            "rot_loss": rot_loss.data,
            "dist_mat_loss": dist_mat_loss.data,
            "sample_size": 1,
            "bsz": 1
        }
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        rot_loss_sum = sum(log.get("rot_loss", 0) for log in logging_outputs)
        dist_mat_loss_sum = sum(log.get("dist_mat_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "rot_loss", rot_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dist_mat_loss", dist_mat_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
       
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return is_train


@register_loss("polyconfm_conf_gen")
class PolyConFMConfGenLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        logging_output = model(**sample["net_input"])
        return 1, 1, logging_output
            

@register_loss("polyconfm_property")
class PolyConFMPropertyLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        reg_output = net_output[0]
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:  
            targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
            targets_std = torch.tensor(self.task.std, device=reg_output.device)
            reg_output = reg_output * targets_std + targets_mean   
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"].view(-1, self.args.num_classes).data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        targets_mean = torch.tensor(self.task.mean, device=targets.device)
        targets_std = torch.tensor(self.task.std, device=targets.device)
        targets = (targets - targets_mean) / targets_std            
        loss = F.mse_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                mae = np.abs(df["predict"] - df["target"]).mean()
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                df = df.groupby("smi").mean()
                agg_mae = np.abs(df["predict"] - df["target"]).mean()
                agg_mse = ((df["predict"] - df["target"]) ** 2).mean()
                agg_r2 = r2_score(df["target"], df["predict"])
                
                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mse", agg_mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_agg_rmse", np.sqrt(agg_mse), sample_size, round=4
                )
                metrics.log_scalar(
                    f"{split}_agg_r2", agg_r2, sample_size, round=4
                )
                metrics.log_scalar(
                    f"{split}_agg_reg", agg_r2 - np.sqrt(agg_mse), sample_size, round=4
                )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return is_train


@register_loss("polyconfm_design")
class PolyConFMDesignLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        loss = model(**sample["net_input"])
        sample_size = sample["net_input"]["src_tokens"].size(0)
        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "bsz": sample["net_input"]["src_tokens"].size(0),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return is_train
    
    
@register_loss("polyconfm_design_inference")
class PolyConFMDesignInferenceLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        logging_output = model(**sample["net_input"])
        return 1, 1, logging_output