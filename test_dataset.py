import torch
from torch.utils.data import DataLoader
import pandas as pd

from dataset import ERA5TWDatasetAurora

def main():
    # Dummy parameters -- replace with actual values as needed
    data_root_dir = "/home/b084020005/era5_tw"
    start_date_hour = pd.Timestamp("2022-01-01 00:00:00")
    end_date_hour = pd.Timestamp("2022-01-05 00:00:00")
    upper_variables = ["u", "v"]  # Replace with your actual variable names
    surface_variables = ["t2m"]   # Replace with your actual variable names
    static_variables = ["lsm", "slt", "z"]  # Replace with your actual static variable names
    levels = [1000, 925, 850, 700, 500, 300, 150, 50]           # Replace with your actual pressure levels
    latitude = (39.75, 5,)           # Replace with your actual latitude range
    longitude = (100, 144.75,)        # Replace with your actual longitude range

    # Instantiate dataset
    dataset = ERA5TWDatasetAurora(
        data_root_dir = data_root_dir,
        start_date_hour = start_date_hour,
        end_date_hour = end_date_hour,
        upper_variables = upper_variables,
        surface_variables = surface_variables,
        static_variables = static_variables,
        levels = levels,
        latitude = latitude,
        longitude = longitude,
        lead_time = 1,
        rollout_step = 6,
    )

    # Wrap in DataLoader
    loader = DataLoader(dataset, batch_size = 4, shuffle = False)

    # Iterate through a few batches
    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        if isinstance(batch, (list, tuple)):
            for j, item in enumerate(batch):
                print(f"  Item {j}: type = {type(item)}")
                if isinstance(item, dict):
                    print(f"    Keys: {list(item.keys())}")
        else:
            print(f"  Batch type: {type(batch)}")
        if i >= 2:
            break

    print(dataset.get_levels())

    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        inputs, outputs, dates = batch
        print(f"{inputs['surf_vars']['2t'].shape=}")
        print(f"{inputs['atmos_vars']['u'].shape=}")
        print(f"{outputs['surf_vars']['2t'].shape=}")
        print(f"{outputs['atmos_vars']['u'].shape=}")
        print(f"{type(dates)=}")
        print(f"{len(dates)=}")
        print(dates)
        break
if __name__ == "__main__":
    main()