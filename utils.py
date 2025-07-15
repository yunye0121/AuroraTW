import torch
from aurora.batch import Batch
from functools import partial
from itertools import product

# version 2.
# def AuroraBatchLoss(
#     pred: Batch, 
#     target: Batch,
#     loss_function: torch.nn.Module,
# ) -> dict:
#     loss_dict = {}
#     total_loss = 0.0
#     var_count = 0

#     # --- Surface Variables ---
#     surf_losses = []
#     loss_dict["surf_vars"] = {}
#     for k in pred.surf_vars:
#         if k not in target.surf_vars:
#             raise KeyError(f"{k} missing in target batch surf_vars.")
#         loss = loss_function(pred.surf_vars[k], target.surf_vars[k])
#         loss_dict["surf_vars"][k] = loss.detach()  # Detach for logging
#         surf_losses.append(loss)
#         var_count += 1

#     # --- Upper Variables ---
#     loss_dict["atmos_vars"] = {}
#     atmos_losses = []
#     levs = pred.metadata.atmos_levels
#     for k in pred.atmos_vars:
#         loss_dict["atmos_vars"][k] = {}
#         if k not in target.atmos_vars:
#             raise KeyError(f"{k} missing in target batch atmos_vars.")
#         # [B, T, L, H, W] -> for each variable, flatten over level
#         pred = pred.atmos_vars[k]   # shape: [B, T, L, H, W]
#         target = target.atmos_vars[k]
#         for i in range(len(levs)):
#             loss = loss_function(pred_flat[:,:,i], target_flat[:,:,i])
#             loss_dict["atmos_vars"][k][levs[i]] = loss.detach()
#             atmos_losses.append(loss)
#             var_count += 1

#     # --- Fast total loss computation ---
#     if surf_losses or atmos_losses:
#         total_loss = torch.stack(surf_losses + atmos_losses).mean()
#     else:
#         total_loss = torch.tensor(0.0, device=next(loss_function.parameters()).device)
#     loss_dict["all_vars"] = total_loss

#     return loss_dict

# version 1.
def AuroraBatchLoss(
    pred: Batch, 
    target: Batch,
    loss_function: torch.nn.Module,
    # reduction: str = "mean"
) -> dict:
    loss_dict = {}
    total_loss = 0.0
    var_count = 0

    # for vars_group in ["surf_vars", "atmos_vars"]:
    loss_dict["surf_vars"] = {}
    
    # get surface / upper variable and all atomosphere variables levels.
    # pred_vars = getattr(pred, vars_group)
    sfc_pred_vars = pred.surf_vars
    # target_vars = getattr(target, vars_group)
    sfc_target_vars = target.surf_vars
    atmos_pred_vars = pred.atmos_vars
    atmos_target_vars = target.atmos_vars
    levs = pred.metadata.atmos_levels
    
    # surface variables: [b, t, h, w]
    # https://microsoft.github.io/aurora/batch.html#batch-surf-vars
    for k in sfc_pred_vars:
        if k not in sfc_target_vars:
            raise KeyError(f"{k} missing in target batch surf_vars.")
        loss = loss_function(
            sfc_pred_vars[k],
            sfc_target_vars[k],
        )
        loss_dict["surf_vars"][k] = loss.detach()
        total_loss += loss
        var_count += 1
    
    loss_dict["atmos_vars"] = {}
    # upper variables: [b, t, l, h, w]
    # https://microsoft.github.io/aurora/batch.html#batch-atmos-vars
    for k in atmos_pred_vars:
        loss_dict["atmos_vars"][k] = {}
        if k not in atmos_target_vars:
            raise KeyError(f"{k} missing in target batch atmos_vars.")
        for i, l in enumerate(levs):
            loss = loss_function(
                atmos_pred_vars[k][:, :, i],
                atmos_target_vars[k][:, :, i],
            )
            loss_dict["atmos_vars"][k][l] = loss.detach()
            total_loss += loss
            var_count += 1

    loss_dict["all_vars"] = (total_loss / var_count) if var_count else 0.0
    return loss_dict

AuroraBatchMAELoss = partial(AuroraBatchLoss, loss_function = torch.nn.L1Loss())
AuroraBatchMSELoss = partial(AuroraBatchLoss, loss_function = torch.nn.MSELoss())

from dataclasses import dataclass
# import torch

@dataclass
class MSEAggregator:
    """
    Implement Error Aggregator, which can also be used for multi-gpu evaluation.
    """
    error_sum: torch.Tensor
    # This is batch count.
    count: int

    def update(self, error_value_tensor: torch.Tensor):
        # mse: a torch scalar or tensor, already reduced over batch/space
        self.error_sum += error_value_tensor.detach()
        self.count += 1

    # def merge(self, other):
    #     self.sum += other.sum
    #     self.count += other.count

    def mean(self):
        if self.count == 0:
            return float('nan')
        return self.error_sum / self.count


#  def map_var_name_for_Aurora(self, var_name: str) -> str:
#         """
#         Map variable names to Aurora's expected names.
#         """
        
#         if var_name in var_name_mapping:
#             return var_name_mapping[var_name]
#         else:
#             return var_name

def prepare_each_lead_time_mse_agg(
    max_lead_time: int,
    surface_variables: list,
    upper_variables: list,
    levels: list,
    device: torch.device
) -> dict:
    agg = {}
    var_name_mapping = {
        "t2m": "2t",
        "u10": "10u",
        "v10": "10v",
        "msl": "msl",
    }
    for t in range(1, max_lead_time + 1):  # usually rollout steps start from 0
        agg[t] = {'surf_vars': {}, 'atmos_vars': {}}
        for var in surface_variables:
            _var = var_name_mapping[var] if var in var_name_mapping else var
            agg[t]['surf_vars'][_var] = MSEAggregator(
                error_sum = torch.zeros(1, device = device),
                count = 0,
            )
        for var in upper_variables:
            _var = var_name_mapping[var] if var in var_name_mapping else var
            agg[t]['atmos_vars'][_var] = {}
            for lev in levels:
                agg[t]['atmos_vars'][_var][lev] = MSEAggregator(
                    error_sum = torch.zeros(1, device = device),
                    count = 0,
                )
    return agg