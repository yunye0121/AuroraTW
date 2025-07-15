#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging
import random
import numpy as np

from aurora import Batch, Metadata
from aurora import rollout
from aurora.model.aurora import AuroraSmall
from dataset import ERA5TWDatasetAurora
from utils import AuroraBatchMAELoss
from utils import prepare_each_lead_time_mse_agg

import xarray as xr
from safetensors.torch import load_file

# =============== Utility Setup ===============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Aurora Evaluation Script (Single GPU)")
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
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
    
    parser.add_argument("--save_folder", type=str, default='./save_files',)
    parser.add_argument("--save_lead_time", type=int, nargs="+", default=None)
    return parser.parse_args()

def create_model(args, device):
    model = AuroraSmall(
        timestep=pd.Timedelta(hours=1),
        use_lora=False,
        autocast=False
    )
    if args.checkpoint_path:
        logger.info(f"Loading checkpoint: {args.checkpoint_path}")
        # state_dict = torch.load(args.checkpoint_path, map_location = "cpu", weights_only = True)
        state_dict = load_file(args.checkpoint_path)
        # model.load_state_dict(state_dict, strict = False)
        model.load_state_dict(state_dict)
    model.to(device)
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

def log_weather_variable_error_with_lead_time(loss_dict, t, lead_time_agg):
    for v in loss_dict["surf_vars"]:
        lead_time_agg[t]["surf_vars"][v].update(loss_dict["surf_vars"][v])
    for v in loss_dict["atmos_vars"]:
        for l in loss_dict["atmos_vars"][v]:
            lead_time_agg[t]["atmos_vars"][v][l].update(loss_dict["atmos_vars"][v][l])

def slice_timeaxis_grouped(labels):
    axis1_length = next(iter(next(iter(labels.values())).values())).shape[1]
    grouped = {}
    for i in range(axis1_length):
        grouped[i] = {}
        for var_type, var_dict in labels.items():
            grouped[i][var_type] = {}
            for var_name, tensor in var_dict.items():
                grouped[i][var_type][var_name] = tensor[:, i:i+1]
    return grouped

def AuroraBatch_2_nc_files(
    batch,
    args,
):
    # surf vars
    surf_vars = batch.surf_vars.keys()
    atmos_vars = batch.atmos_vars.keys()
    static_vars = batch.static_vars.keys()
    print(surf_vars)
    print(atmos_vars)
    print(static_vars)

    def _np(d):
        return d.detach().cpu().numpy()

    for surf_var in surf_vars:
        print(batch.surf_vars[surf_var].shape)
    for atmos_var in atmos_vars:
        print(batch.atmos_vars[atmos_var].shape)
    for static_var in static_vars:
        print(batch.static_vars[static_var].shape)

    _s = set(
        [batch.surf_vars[var].shape[0] for var in surf_vars] +
        [batch.atmos_vars[var].shape[0] for var in atmos_vars]
    )

    print(_s)
    assert len(_s) == 1

    batch_dim = next(iter(_s))
    # print(batch_dim_num)

    # assert len(
    #     set(
    #         [batch.surf_vars[var].shape[0] for var in surf_vars] +
    #         [batch.atmos_vars[var].shape[0] for var in atmos_vars]
    #     )
    # ) == 1
    for i in range(batch_dim):
        # Prepare variables with batch dimension sliced
        data_vars = {}

        # Surf vars: drop batch dim
        for k, v in batch.surf_vars.items():
            arr = _np(v)[i]  # shape: (history, lat, lon)
            data_vars[f"surf_{k}"] = (("history", "latitude", "longitude"), arr)

        # Atmos vars: drop batch dim
        for k, v in batch.atmos_vars.items():
            arr = _np(v)[i]  # shape: (history, level, lat, lon)
            data_vars[f"atmos_{k}"] = (("history", "level", "latitude", "longitude"), arr)

        # Static vars: no batch dim
        for k, v in batch.static_vars.items():
            arr = _np(v)
            data_vars[f"static_{k}"] = (("latitude", "longitude"), arr)

        # Metadata may need to be indexed or used as is
        coords = {
            "latitude": _np(batch.metadata.lat),
            "longitude": _np(batch.metadata.lon),
            "time": [batch.metadata.time[i]],  # just this batch element's time
            "level": list(batch.metadata.atmos_levels),
            "rollout_step": batch.metadata.rollout_step,
        }

        ds = xr.Dataset(data_vars, coords = coords)
        # out_path = f"batch_{i}.nc"
        hours = int(batch.metadata.rollout_step)
        output_file_name = f"{(batch.metadata.time[i] - pd.Timedelta(hours=hours)).strftime('%Y%m%d_%H%M%S')}+{hours}hr.nc"
        # output_file_name = f"{(batch.metadata.time[i] - pd.Timedelta(hours = batch.metadata.rollout_step)).strftime('%Y%m%d_%H%M%S')}+{pd.Timedelta(hours = f"{int(batch.metadata.rollout_step)}" )}hr.nc"
        output_path = os.path.join(args.save_folder, output_file_name)
        ds.to_netcdf( output_path )
        print(f"Saved {output_path}")


def evaluate(args, model, dataloader, criterion, device, lead_time_mse_agg):
    model.eval()
    latitudes, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels, dates = batch
            # Move tensors to device
            for k1 in inputs:
                for k2 in inputs[k1]:
                    inputs[k1][k2] = inputs[k1][k2].to(device)
            for k1 in labels:
                for k2 in labels[k1]:
                    labels[k1][k2] = labels[k1][k2].to(device)
            # (If static_data['static_vars'] is tensor, move to device)
            if isinstance(static_data["static_vars"], torch.Tensor):
                static_data["static_vars"] = static_data["static_vars"].to(device)

            new_label = slice_timeaxis_grouped(labels)

            # Mixed precision is optional, here just standard
            _input = Batch(
                surf_vars=inputs["surf_vars"],
                atmos_vars=inputs["atmos_vars"],
                static_vars=static_data["static_vars"],
                metadata=Metadata(
                    lat=latitudes,
                    lon=longitude,
                    time=tuple(map(lambda d: pd.Timestamp(d), dates)),
                    atmos_levels=levels,
                ),
            )
            _ar_preds = [pred for pred in rollout(model, _input, steps=args.rollout_step)]

            for t in range(1, args.rollout_step + 1):
                _pred = _ar_preds[t - 1]
                _label = Batch(
                    surf_vars=new_label[t - 1]["surf_vars"],
                    atmos_vars=new_label[t - 1]["atmos_vars"],
                    static_vars=static_data["static_vars"],
                    metadata=Metadata(
                        lat=latitudes,
                        lon=longitude,
                        time=tuple(map(lambda d: pd.Timestamp(d), dates)),
                        atmos_levels=levels,
                    ),
                )

                loss_dict = criterion(_pred, _label)
                log_weather_variable_error_with_lead_time(
                    loss_dict,
                    t,
                    lead_time_mse_agg,
                )

                if args.save_lead_time and t in args.save_lead_time:
                    # _pred.to_netcdf(
                    #     path=f"{args.save_folder}/rs{_pred.metadata.rollout_step}_dt{min(_pred.metadata.time)}-{max(_pred.metadata.time)}.nc"
                    # )
                    AuroraBatch_2_nc_files(
                        batch = _pred,
                        args = args
                    )
            break

def export_agg_to_csv_by_level(
    lead_time_mse_agg,
    out_path = "evaluation_results.csv",
    ):

    lead_times = sorted(lead_time_mse_agg.keys())
    lead_time_labels = [f"{t}h" for t in lead_times]

    surf_vars = set()
    atmos_vars_levels = dict()
    for t in lead_time_mse_agg:
        for var in lead_time_mse_agg[t]["surf_vars"]:
            surf_vars.add(var)
        for var in lead_time_mse_agg[t]["atmos_vars"]:
            if var not in atmos_vars_levels:
                atmos_vars_levels[var] = set()
            for lev in lead_time_mse_agg[t]["atmos_vars"][var]:
                atmos_vars_levels[var].add(lev)
    surf_vars = sorted(list(surf_vars))

    atmos_rows = []
    for var in sorted(atmos_vars_levels.keys()):
        levels = sorted(list(atmos_vars_levels[var]), reverse=True)
        for lev in levels:
            atmos_rows.append((var, lev))

    rows = []
    row_names = []

    for var in surf_vars:
        row = []
        for t in lead_times:
            agg = lead_time_mse_agg[t]["surf_vars"].get(var)
            row.append( np.sqrt(agg.mean()).item() if agg is not None else None)
        rows.append(row)
        row_names.append(var)

    for var, lev in atmos_rows:
        row = []
        for t in lead_times:
            agg = lead_time_mse_agg[t]["atmos_vars"].get(var, {}).get(lev)
            row.append( np.sqrt(agg.mean().item()) if agg is not None else None)
        rows.append(row)
        row_names.append(f"{var}_{lev}")

    df = pd.DataFrame(rows, index=row_names, columns=lead_time_labels)
    df.to_csv(out_path)
    print(f"Saved CSV to {out_path}")
    return df

def main():
    args = parse_args()
    set_seed(args.seed)
    logger.info("Running single-GPU evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args, device)
    dataset = create_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    criterion = AuroraBatchMAELoss

    if args.save_lead_time is not None:
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        logger.info(f"Saving lead time outputs to {args.save_folder}")

    lead_time_mse_agg = prepare_each_lead_time_mse_agg(
        max_lead_time=args.rollout_step,
        surface_variables=args.surface_variables,
        upper_variables=args.upper_variables,
        levels=args.levels,
        device=device,
    )

    evaluate(
        args,
        model,
        dataloader,
        criterion,
        device,
        lead_time_mse_agg
    )

    # Print results
    # for t in lead_time_mse_agg:
    #     for var_group in lead_time_mse_agg[t]:
    #         if var_group == "surf_vars":
    #             for var in lead_time_mse_agg[t][var_group]:
    #                 agg = lead_time_mse_agg[t][var_group][var]
    #                 print(f"Lead {t} | {var_group} | {var} | MSE: {agg.mean().item():.6e} | Count: {agg.count}")
    #         elif var_group == "atmos_vars":
    #             for var in lead_time_mse_agg[t][var_group]:
    #                 for lev in lead_time_mse_agg[t][var_group][var]:
    #                     agg = lead_time_mse_agg[t][var_group][var][lev]
    #                     print(f"Lead {t} | {var_group} | {var} | Level {lev} | MSE: {agg.mean().item():.6e} | Count: {agg.count}")

    # export_agg_to_csv_by_level(lead_time_mse_agg, out_path = "evaluation_results.csv")

if __name__ == "__main__":
    main()
