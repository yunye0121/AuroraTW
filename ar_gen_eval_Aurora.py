#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from aurora import Batch, Metadata
from aurora import rollout
from aurora.model.aurora import AuroraSmall
from dataset import ERA5TWDatasetAurora
from utils import AuroraBatchMAELoss
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from utils import prepare_each_lead_time_mse_agg

from tqdm.auto import tqdm

logger = get_logger(__name__, log_level = "INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Aurora Evaluation Script")
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start_date_hour', type=str, required=True)
    parser.add_argument('--end_date_hour', type=str, required=True)
    parser.add_argument('--upper_variables', type=str, nargs='+', required=True)
    parser.add_argument('--surface_variables', type=str, nargs='+', required=True)
    parser.add_argument('--static_variables', type=str, nargs='+', required=True)
    parser.add_argument('--levels', type=int, nargs='+', required=True)
    parser.add_argument('--latitude', type=float, nargs=2, required=True)
    parser.add_argument('--longitude', type=float, nargs=2, required=True)
    parser.add_argument('--lead_time', type=int, default=0)
    parser.add_argument('--rollout_step', type=int, default=1)

    parser.add_argument("--save_lead_time", type = int, nargs = "+", default = None,)

    return parser.parse_args()

def create_model(args):
    model = AuroraSmall(
        timestep = pd.Timedelta(hours = 1),
        use_lora = False,
        autocast = False
    )
    if args.checkpoint_path:
        logger.info(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location = "cpu")
        model.load_state_dict(state_dict, strict = False)
    #     accelerator.load_state(checkpoint_path)
    model.eval()
    return model

def create_dataset(args):
    ds = ERA5TWDatasetAurora(
        data_root_dir=args.data_root_dir,
        start_date_hour=args.start_date_hour,
        end_date_hour=args.end_date_hour,
        upper_variables=args.upper_variables,
        surface_variables=args.surface_variables,
        static_variables=args.static_variables,
        levels=args.levels,
        latitude=args.latitude,
        longitude=args.longitude,
        lead_time=args.lead_time,
        rollout_step=args.rollout_step,
    )
    return ds

def log_weather_variable_error_with_lead_time(
        loss_dict: dict,
        t: int,
        lead_time_agg: dict,
    ):
    
    # Iterate surface_far.
    for v in loss_dict["surf_vars"]:
        # print(f'{loss_dict["surf_vars"]=}')
        # Update the corresponding error aggregator ( note that loss_dict[g][v] has already averaged by batch num. )
        lead_time_agg[t]["surf_vars"][v].update( loss_dict["surf_vars"][v] )
    for v in loss_dict["atmos_vars"]:
        # print(f'{loss_dict["atmos_vars"]=}')
        for l in loss_dict["atmos_vars"][v]:
            lead_time_agg[t]["atmos_vars"][v][l].update( loss_dict["atmos_vars"][v][l] )

def slice_timeaxis_grouped(labels):
    # Get the max axis-1 length across all variables (assuming all variables in a category have the same axis-1 size)
    # You can adapt if that's not the case
    axis1_length = next(iter(next(iter(labels.values())).values())).shape[1]

    # Build the new structure
    grouped = {}
    for i in range(axis1_length):
        grouped[i] = {}
        for var_type, var_dict in labels.items():
            grouped[i][var_type] = {}
            for var_name, tensor in var_dict.items():
                grouped[i][var_type][var_name] = tensor[:, i:i+1]
    return grouped


# def save_AuroraBatch_2_nc(
#     AuroraBatch: Batch,
#     flle_path: str,
# ):
#     """
#     We need to visulize the predicition into single sample instead of batch.
#     """
#     AuroraBatch.to_netcdf(
#         path = flle_path,
#     )

def evaluate(args, model, dataloader, criterion, accelerator, lead_time_mse_agg):
    model.eval()
    # total_loss = 0.0
    latitudes, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    with torch.inference_mode():
        pbar = tqdm(dataloader, disable = not accelerator.is_local_main_process, desc="Evaluating")
        for batch in pbar:
            inputs, labels, dates = batch
            new_label = slice_timeaxis_grouped(labels)
            # print(f"{labels['surf_vars']['2t'].shape=}")
            # print(f"{labels['atmos_vars']['u'].shape=}")
            # print(f"{slice_timeaxis_grouped(labels)[0]['surf_vars']['2t'].shape=}")
            # print(f"{slice_timeaxis_grouped(labels)[0]['atmos_vars']['u'].shape=}")
            # print(f"{dates=}")

            with accelerator.autocast():
                _input = Batch(
                    surf_vars = inputs["surf_vars"],
                    atmos_vars = inputs["atmos_vars"],
                    static_vars = static_data["static_vars"],
                    metadata = Metadata(
                        lat = latitudes,
                        lon = longitude,
                        time = tuple(map(lambda d: pd.Timestamp(d), dates)),
                        atmos_levels = levels,
                    ),
                )
                # _label = Batch(
                #     surf_vars = labels["surf_vars"],
                #     atmos_vars = labels["atmos_vars"],
                #     static_vars = static_data["static_vars"],
                #     metadata = Metadata(
                #         lat = latitudes,
                #         lon = longitude,
                #         time = tuple(map(lambda d: pd.Timestamp(d), dates)),
                #         atmos_levels = levels,
                #     ),
                # )
                
                _ar_preds = [_pred for _pred in rollout(model, _input, steps = args.rollout_step)]
                
                assert len(_ar_preds) == args.rollout_step

                for t in range(1, args.rollout_step + 1):
                    
                    # Select spefic timestamp.
                    _pred = _ar_preds[t - 1]

                    _label = Batch(
                        surf_vars = new_label[t - 1]["surf_vars"],
                        atmos_vars = new_label[t - 1]["atmos_vars"],
                        static_vars = static_data["static_vars"],
                        metadata = Metadata(
                            lat = latitudes,
                            lon = longitude,
                            time = tuple(map(lambda d: pd.Timestamp(d), dates)),
                            atmos_levels = levels,
                        ),
                    )

                    # Note here Batch object in unormalized by default.
                    loss_dict = criterion(_pred, _label)
                    log_weather_variable_error_with_lead_time(
                        loss_dict,
                        t,
                        lead_time_mse_agg,
                    )

                    if args.save_lead_time and t in args.save_lead_time:
                        _pred.to_netcdf(
                            path = f"{_pred.metadata.rollout_step}_{min(_pred.metadata.time)} - {max(_pred.metadata.time)}.nc"
                        )

                # break          
                del inputs, labels, dates, _input, _label, _ar_preds, _pred, new_label,
                torch.cuda.empty_cache()

    for t in lead_time_mse_agg:
        for var_group in lead_time_mse_agg[t]:
            for var in lead_time_mse_agg[t][var_group]:
                if var_group == "surf_vars":
                    _agg = lead_time_mse_agg[t][var_group][var]
                    # Reduce sum and count.
                    _agg.error_sum = accelerator.reduce(
                        _agg.error_sum, reduction = "sum"
                    )
                    _agg.count = accelerator.reduce(
                        torch.tensor(
                            _agg.count, device = _agg.error_sum.device),
                            reduction = "sum",
                    ).item()
                elif var_group == "atmos_vars":
                    for lev in lead_time_mse_agg[t][var_group][var]:
                        _agg = lead_time_mse_agg[t][var_group][var][lev]
                        # Reduce sum and count.
                        _agg.error_sum = accelerator.reduce(
                            _agg.error_sum, reduction = "sum"
                        )
                        _agg.count = accelerator.reduce(
                            torch.tensor(
                                _agg.count, device = _agg.error_sum.device),
                                reduction = "sum",
                        ).item()


import pandas as pd

def export_agg_to_csv_by_level(lead_time_mse_agg, out_path="evaluation_results.csv"):
    # Figure out lead times
    lead_times = sorted(lead_time_mse_agg.keys())
    lead_time_labels = [f"{t}h" for t in lead_times]

    # 1. Collect surface and atmospheric variable/levels
    surf_vars = set()
    atmos_vars_levels = dict()  # {var: set(levels)}
    for t in lead_time_mse_agg:
        for var in lead_time_mse_agg[t]["surf_vars"]:
            surf_vars.add(var)
        for var in lead_time_mse_agg[t]["atmos_vars"]:
            if var not in atmos_vars_levels:
                atmos_vars_levels[var] = set()
            for lev in lead_time_mse_agg[t]["atmos_vars"][var]:
                atmos_vars_levels[var].add(lev)
    surf_vars = sorted(list(surf_vars))  # Sort surface vars alphabetically

    # Sort atmospheric vars by variable, and levels descending
    atmos_rows = []
    for var in sorted(atmos_vars_levels.keys()):
        levels = sorted(list(atmos_vars_levels[var]), reverse=True)  # Descending order
        for lev in levels:
            atmos_rows.append((var, lev))

    # 2. Fill the data for each row/lead time
    rows = []
    row_names = []

    # Surface vars first
    for var in surf_vars:
        row = []
        for t in lead_times:
            agg = lead_time_mse_agg[t]["surf_vars"].get(var)
            row.append(agg.mean().item() if agg is not None else None)
        rows.append(row)
        row_names.append(var)

    # Atmospheric vars (by variable, by level descending)
    for var, lev in atmos_rows:
        row = []
        for t in lead_times:
            agg = lead_time_mse_agg[t]["atmos_vars"].get(var, {}).get(lev)
            row.append(agg.mean().item() if agg is not None else None)
        rows.append(row)
        row_names.append(f"{var}_{lev}")

    # 3. Create DataFrame and save
    df = pd.DataFrame(rows, index=row_names, columns=lead_time_labels)
    df.to_csv(out_path)
    print(f"Saved CSV to {out_path}")
    return df

def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    logger.info(accelerator.state)
    model = create_model(args)
    dataset = create_dataset(args)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = True)
    criterion = AuroraBatchMAELoss
    # model_stats = model.surf_stats
    model, dataloader = accelerator.prepare(
        model, dataloader
    )

    # if args.checkpoint_path:
    #     accelerator.load_state(args.checkpoint_path)

    model = accelerator.unwrap_model(model)

    lead_time_mse_agg = prepare_each_lead_time_mse_agg(
        max_lead_time = args.rollout_step,
        surface_variables = args.surface_variables,
        upper_variables = args.upper_variables,
        levels = args.levels,
        # device = model.device,
        device = next(model.parameters()).device,
    )

    evaluate(
        args,
        model,
        dataloader,
        criterion,
        accelerator,
        lead_time_mse_agg
    )

    if accelerator.is_main_process:
        for t in lead_time_mse_agg:
            for var_group in lead_time_mse_agg[t]:
                if var_group == "surf_vars":
                    for var in lead_time_mse_agg[t][var_group]:
                        agg = lead_time_mse_agg[t][var_group][var]
                        print(f"Lead {t} | {var_group} | {var} | MSE: {agg.mean().item():.6e} | Count: {agg.count}")
                elif var_group == "atmos_vars":
                    for var in lead_time_mse_agg[t][var_group]:
                        for lev in lead_time_mse_agg[t][var_group][var]:
                            agg = lead_time_mse_agg[t][var_group][var][lev]
                            print(f"Lead {t} | {var_group} | {var} | Level {lev} | MSE: {agg.mean().item():.6e} | Count: {agg.count}")

        # Export results to CSV
        export_agg_to_csv_by_level(lead_time_mse_agg, out_path="evaluation_results.csv")
        

if __name__ == "__main__":
    main()
