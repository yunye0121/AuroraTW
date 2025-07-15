#!/usr/bin/env python
# coding=utf-8

import argparse
import pandas as pd
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# -- Your custom modules here --
from aurora import Batch, Metadata
from aurora.model.aurora import AuroraSmall
# from aurora.utils.loss import MAE
# from aurora.utils.metrics import my_rmse_val
# from dataset import Raw_ERA5

import shutil
from dataset import ERA5TWDatasetAurora
from utils import AuroraBatchMAELoss
from tqdm.auto import tqdm

logger = get_logger(__name__, log_level = "INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Aurora Training Script (HF Style)")
    parser.add_argument('--data_root_dir', type=str, required=True, help="Root dir of the dataset")
    parser.add_argument('--leadtime', type=int, required=True, help="Time interval between input[previous, current]")
    parser.add_argument('--rollout_step', type=int, required=True, help="Rollout step")
    # parser.add_argument('--train_split', type=str, default="train_Sub", help="Dataset split (default: train_Sub)")
    parser.add_argument('--use_pretrained_weight', action='store_true', help="Use pretrained weights from Aurora")
    parser.add_argument('--epochs', type=int, default=5, help="Training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--train_batch_size', type=int, default=16, help="Batch size per device")
    parser.add_argument('--val_batch_size', type=int, default=16, help="Batch size per device")
    parser.add_argument('--num_workers', type=int, default=4, help="Dataloader worker count")
    parser.add_argument('--ckpt_prefix', type=str, default="", help="Prefix for output files")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to model checkpoint")
    # parser.add_argument('--mode', type=str, choices=["train_val_test", "resume", "only_test"], required=True)
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--train_start_date_hour', type=str, required=True, help="Start date in 'YYYY-MM-DD HH:MM:SS' format")
    parser.add_argument('--train_end_date_hour', type=str, required=True, help="End date in 'YYYY-MM-DD HH:MM:SS' format")
    parser.add_argument('--val_start_date_hour', type=str, required=True, help="Start date in 'YYYY-MM-DD HH:MM:SS' format")
    parser.add_argument('--val_end_date_hour', type=str, required=True, help="End date in 'YYYY-MM-DD HH:MM:SS' format")
    parser.add_argument('--upper_variables', type=str, nargs='+', required=True, help="List of upper atmosphere variables")    
    parser.add_argument('--surface_variables', type=str, nargs='+', required=True, help="List of surface variables")
    parser.add_argument('--static_variables', type=str, nargs='+', required=True, help="List of static variables")
    parser.add_argument('--levels', type=int, nargs='+', required=True, help="List of pressure levels")
    parser.add_argument('--latitude', type=float, nargs=2, required=True, help="Latitude range as two floats (min, max)")
    parser.add_argument('--longitude', type=float, nargs=2, required=True, help="Longitude range as two floats (min, max)")
    parser.add_argument('--lead_time', type=int, default=0, help="Lead time for the dataset")
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="auroatw",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument('--checkpointing_epochs', type=int, default=5, help="Checkpointing frequency in epochs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
        )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="AuroraTW",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    return parser.parse_args()

def create_model(args):
    model = AuroraSmall(use_lora = False, autocast = False, timestep = pd.Timedelta(hours = 1))
    if args.use_pretrained_weight:
        logger.info("Loading pretrained weights...")
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt", strict = False)
    elif args.checkpoint_path:
        logger.info(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location = "cpu")
        model.load_state_dict(state_dict, strict = False)
    return model


def create_dataset(args, split):
    if split == "train":
        ds = ERA5TWDatasetAurora(
                data_root_dir = args.data_root_dir,
                start_date_hour = args.train_start_date_hour,
                end_date_hour = args.train_end_date_hour,
                upper_variables = args.upper_variables,
                surface_variables = args.surface_variables,
                static_variables = args.static_variables,
                levels = args.levels,
                latitude = args.latitude,
                longitude = args.longitude,
                lead_time = args.lead_time,
                rollout_step = args.rollout_step,
            )
    elif split == "val":
        ds = ERA5TWDatasetAurora(
                data_root_dir = args.data_root_dir,
                start_date_hour = args.val_start_date_hour,
                end_date_hour = args.val_end_date_hour,
                upper_variables = args.upper_variables,
                surface_variables = args.surface_variables,
                static_variables = args.static_variables,
                levels = args.levels,
                latitude = args.latitude,
                longitude = args.longitude,
                lead_time = args.lead_time,
                rollout_step = args.rollout_step,
            )
    else:
        raise Exception("Do not support this dataset split!")
    return ds

# def collate_fn(batch):
#     return batch  # Adjust as needed for your data structure

def save_checkpoint_by_epoch(args, accelerator, output_dir, epoch):

    if args.checkpointing_epochs > 0 and epoch % args.checkpointing_epochs == 0:
        if accelerator.is_main_process:
            # List all checkpoint dirs
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            
            # If too many, remove oldest
            if args.checkpoints_total_limit is not None:
                if len(checkpoints) >= args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[:num_to_remove]
                    for removing_checkpoint in removing_checkpoints:
                        removing_path = os.path.join(output_dir, removing_checkpoint)
                        shutil.rmtree(removing_path)
                        logger.info(f"Removed old checkpoint: {removing_path}")

            save_path = os.path.join(output_dir, f"checkpoint-{epoch}")
            os.makedirs(save_path, exist_ok = True)
            accelerator.save_state(save_path)  # saves model, optimizer, etc.

            logger.info(f"Saved checkpoint to {save_path}")

# def save_checkpoint_by_epoch(args, accelerator, output_dir, epoch, model, optimizer=None):

#     if args.checkpointing_epochs > 0 and epoch % args.checkpointing_epochs == 0:
#         if accelerator.is_main_process:
#             # List all checkpoint files
#             checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
#             # checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
#             checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))

#             # Remove oldest if exceeding limit
#             if args.checkpoints_total_limit is not None and len(checkpoints) >= args.checkpoints_total_limit:
#                 num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
#                 removing_checkpoints = checkpoints[:num_to_remove]
#                 for removing_checkpoint in removing_checkpoints:
#                     removing_path = os.path.join(output_dir, removing_checkpoint)
#                     if os.path.isdir(removing_path):
#                         shutil.rmtree(removing_path)
#                     elif os.path.isfile(removing_path):
#                         os.remove(removing_path)
#                     logger.info(f"Removed old checkpoint: {removing_path}")

#             save_path = os.path.join(output_dir, f"checkpoint-{epoch}.ckpt")
#             to_save = accelerator.unwrap_model(model).state_dict()
#             checkpoint = {"epoch": epoch, "model_state_dict": to_save}
#             if optimizer is not None:
#                 checkpoint["optimizer_state_dict"] = optimizer.state_dict()
#             torch.save(checkpoint, save_path)
#             logger.info(f"Saved checkpoint to {save_path}")


def train_epoch(
        args,
        model,
        dataloader,
        optimizer,
        criterion,
        accelerator,
        epoch,
        model_stats,
        scaler = None,
    ):
    model.train()
    total_train_loss = 0.0

    latitudes, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()


    # print(f"Training on latitudes: {latitudes}, longitude: {longitude}, levels: {levels}")

    pbar = tqdm(
        dataloader, 
        disable = not accelerator.is_local_main_process, 
        desc=f"train_epoch: {epoch}",
        # ncols = 120
    )
    for batch in pbar:
        train_input, train_label, train_dates = batch

        # print(f"Model device: {next(model.parameters()).device}")
        # print(f"surf_vars device: {train_input['surf_vars']['2t'].device}")
        # print(f"atmos_vars device: {train_input['atmos_vars']['u'].device}")
        # print(f"train_label surf_vars device: {train_label['surf_vars']['2t'].device}")
        # print(f"train_label atmos_vars device: {train_label['atmos_vars']['u'].device}")

        optimizer.zero_grad()
        with accelerator.autocast():
            
            _input = Batch(
                surf_vars = train_input["surf_vars"],
                atmos_vars = train_input["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = latitudes,
                    lon = longitude,
                    time = tuple(map(lambda d: pd.Timestamp(d), train_dates)),
                    atmos_levels = levels,
                ),
            )

            _label = Batch(
                surf_vars = train_label['surf_vars'],
                atmos_vars = train_label['atmos_vars'],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = latitudes,
                    lon = longitude,
                    time = tuple(map(lambda d: pd.Timestamp(d), train_dates)),
                    atmos_levels = levels,
                ),
            )

            _pred = model.forward(_input)


            # print(f"pred surf_var device: {_pred.surf_vars['2t'].device}")
            # print(f"pred atmos_var device: {_pred.atmos_vars['u'].device}")
            # train_label = train_label.normalise(surf_stats=model.surf_stats)
            loss_dict = criterion(
                _pred.normalise(surf_stats = model_stats),
                _label.normalise(surf_stats = model_stats)
            )
            
            loss = loss_dict["all_vars"]
        
        accelerator.backward(loss)
        optimizer.step()
        # dist_avg_train_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        dist_avg_train_loss = accelerator.gather(loss).mean()
        # dist_avg_train_loss = accelerator.gather(loss)
        # print(f"{dist_avg_train_loss.shape=}")
        # total_loss += loss.item()
        total_train_loss += dist_avg_train_loss.item()
        # Optionally: add logging here
        if accelerator.is_local_main_process:
            pbar.set_postfix({"train_step_loss": f"{dist_avg_train_loss.item():.8f}"})
        # del train_input, train_label, _input, _label, _pred, loss

    train_epoch_loss = total_train_loss / len(dataloader)
    # logger.info(f"Epoch {epoch + 1} - train epoch Loss: {train_epoch_loss:.8f}")
    return train_epoch_loss

def val_epoch(
        args,
        model,
        dataloader,
        criterion,
        accelerator,
        epoch,
        model_stats,
    ):
    model.eval()
    total_val_loss = 0.0

    latitudes, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    pbar = tqdm(
        dataloader, 
        disable = not accelerator.is_local_main_process, 
        desc=f"val_epoch: {epoch}",
        # ncols = 120
    )

    with torch.inference_mode():
        for batch in pbar:
            val_input, val_label, val_dates = batch
            # print(f"surf_vars device: {val_input['surf_vars']['2t'].device}")
            # print(f"atmos_vars device: {val_input['atmos_vars']['u'].device}")
            # print(f"val_label surf_vars device: {val_label['surf_vars']['2t'].device}")
            # print(f"val_label atmos_vars device: {val_label['atmos_vars']['u'].device}")
            with accelerator.autocast():
                _input = Batch(
                    surf_vars = val_input["surf_vars"],
                    atmos_vars = val_input["atmos_vars"],
                    static_vars = static_data["static_vars"],
                    metadata = Metadata(
                        lat = latitudes,
                        lon = longitude,
                        time = tuple(map(lambda d: pd.Timestamp(d), val_dates)),
                        atmos_levels = levels,
                    ),
                )

                _label = Batch(
                    surf_vars = val_label['surf_vars'],
                    atmos_vars = val_label['atmos_vars'],
                    static_vars = static_data["static_vars"],
                    metadata = Metadata(
                        lat = latitudes,
                        lon = longitude,
                        time = tuple(map(lambda d: pd.Timestamp(d), val_dates)),
                        atmos_levels = levels,
                    ),
                )
                # pred = model.forward(val_input)
                _pred = model.forward(_input)
                # val_label = val_label.normalise(surf_stats=model.surf_stats)
                # loss = criterion(pred, val_label)                
                loss_dict = criterion(_pred.normalise(surf_stats = model_stats), _label.normalise(surf_stats = model_stats))
                loss = loss_dict["all_vars"]
            
            # dist_avg_val_loss = accelerator.gather(loss.repeat(args.val_batch_size)).mean()
            dist_avg_val_loss = accelerator.gather(loss).mean()
            # print(f"{loss.shape=}")
            
            total_val_loss += dist_avg_val_loss.item()
            # total_val_loss += loss.item()
            if accelerator.is_local_main_process:
                pbar.set_postfix({"step_loss": f"{dist_avg_val_loss.item():.8f}"})
            
            # del val_input, val_label, _pred, loss

    epoch_val_loss = total_val_loss / len(dataloader)
    # logger.info(f"Epoch {epoch+1} - {split} Loss: {avg_loss:.4f}")
    return epoch_val_loss

# def test(model, dataloader, accelerator):
#     model.eval()
#     metrics_per_step = []  # Store per-step metrics if needed
#     with torch.inference_mode():
#         for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
#             test_input, test_label = batch[0], batch[1]
#             pred = model.forward(test_input)
#             # Here you can call your own metric functions, e.g.:
#             # metric = my_rmse_val(pred, test_label, ...)
#             # metrics_per_step.append(metric)
#             del test_input, test_label, pred
#     logger.info("Testing complete.")
#     # Optionally: save metrics to file, etc.

def main():
    args = parse_args()
    set_seed(args.seed)
    # accelerator = Accelerator()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tracker_config.pop("validation_prompts")
        # print(f"{type(args.tracker_project_name)=}")
        # print(f"{type(tracker_config)=}")
        accelerator.init_trackers(args.tracker_project_name, config = tracker_config)
        
    logger.info(accelerator.state)

    # Directory setup
    ckpt_dir = os.path.join(args.output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok = True)

    # Model, data, optimizer, loss
    model = create_model(args)
    train_dataset = create_dataset(args, "train")
    val_dataset = create_dataset(args, "val")

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle = True,
        num_workers = args.num_workers, pin_memory = True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle = False,
        num_workers = args.num_workers, pin_memory = True,
    )
    # val_loader = DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn
    # )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn
    # )

    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = 1e-4)
    criterion = AuroraBatchMAELoss
    model_stats = model.surf_stats

    # Prepare with accelerator (moves to device, handles DDP if needed)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # print(f"{model.surf_stats=}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(args, model, train_loader, optimizer, criterion, accelerator, epoch, model_stats)
        val_loss = val_epoch(args, model, val_loader, criterion, accelerator, epoch, model_stats)
        if accelerator.is_local_main_process:
            print(f"epoch {epoch} - train Loss: {train_loss:.8f}")
            accelerator.log( {"train_epoch_loss": train_loss}, step = epoch)
            print(f"epoch {epoch} - val Loss: {val_loss:.8f}")
            accelerator.log( {"val_epoch_loss": val_loss}, step = epoch)
            save_checkpoint_by_epoch(
                # args, accelerator, ckpt_dir, epoch, model, optimizer,
                args, accelerator, ckpt_dir, epoch,
            )
        accelerator.wait_for_everyone()
 
    # Training/validation loop
    # if args.mode in ["train_val_test", "resume"]:
    #     best_val_loss = float("inf")
        # for epoch in range(args.epochs):
        #     train_loss = train_epoch(model, train_loader, optimizer, criterion, accelerator, epoch)
            # val_loss = evaluate(model, val_loader, criterion, accelerator, epoch)
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     save_checkpoint(model, output_dir, epoch, args.ckpt_prefix)

    # Test/inference loop
    # if args.mode in ["train_val_test", "only_test"]:
    #     logger.info("Starting test evaluation...")
    #     test(model, test_loader, accelerator)

if __name__ == "__main__":
    main()
