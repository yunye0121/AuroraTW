from math import ceil
from pathlib import Path
import glob

import pandas as pd
import torch
import torch.utils.data as D
import xarray as xr
from einops import pack, rearrange

class ERA5TWDatasetAurora(D.Dataset):
    """
    """
    def __init__(
        self,
        data_root_dir: str,
        start_date_hour: pd.Timestamp,
        end_date_hour: pd.Timestamp,
        upper_variables: list[str],
        surface_variables: list[str],
        static_variables: list[str],
        levels: list[int],
        latitude: tuple[int, int],
        longitude: tuple[int, int],
        lead_time: int = 0,
        rollout_step: int = 1,
        # step_size: int = 1,
        # seq_len: int = 2,
        # flatten: bool = True,
        # standardization: bool = False,
        # stat_dict_path: str = None,
        # get_stat: bool = False,
        get_datetime: bool = True,
    ) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.start_date_hour = pd.Timestamp(start_date_hour)
        self.end_date_hour = pd.Timestamp(end_date_hour)
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.static_variables = static_variables
        self.levels = levels
        self.lead_time = lead_time
        self.rollout_step = rollout_step
        self.latitude = latitude
        self.longitude = longitude
        # self.step_size = step_size
        # self.flatten = flatten
        # self.seq_len = seq_len
        # self.standardization = standardization
        # self.stat_dict_path = stat_dict_path
        # if self.standardization:
        #     self.upper_mean, self.upper_std, self.surface_mean, self.surface_std = self._load_stat()

        # self.get_stat = self.standardization and get_stat
        self.get_datetime = get_datetime
    
    # def standardize(self, _data, mean, std):
    #     """
    #     Standardize data using mean and std.
    #     For sequence data, broadcasting will handle the time dimension.
    #     """
    #     # For sequence data, mean and std won't have the same shape,
    #     # as _data has an extra time dimension at the front
    #     return (_data - mean) / std

    # def _load_stat(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     assert self.stat_dict_path is not None

    #     stat = torch.load(self.stat_dict_path, weights_only=True)
    #     upper_mean, _ = pack([stat["mean_upper"][var] for var in self.upper_variables], "* p h w")
    #     upper_std, _ = pack([stat["std_upper"][var] for var in self.upper_variables], "* p h w")
    #     surface_mean, _ = pack([stat["mean_surface"][var] for var in self.surface_variables], "* h w")
    #     surface_std, _ = pack([stat["std_surface"][var] for var in self.surface_variables], "* h w")

    #     return upper_mean, upper_std, surface_mean, surface_std 

    # def _stack_nc(self, upper_nc, surface_nc) -> tuple[torch.Tensor, torch.Tensor]:
    #     upper_data, _ = pack(
    #         [rearrange(upper_nc[v].values, "() l h w -> l h w")
    #         for v in self.upper_variables], "* l h w"
    #     )
    #     upper_data = torch.Tensor(upper_data)
    #     surface_data, _ = pack(
    #         [rearrange(surface_nc[v].values, "() h w -> h w")
    #         for v in self.surface_variables], "* h w"
    #     )
    #     surface_data = torch.Tensor(surface_data)

    #     return upper_data, surface_data

    # def flatten_var_seq(self, upper_data, surface_data):
    #     """
    #     Flatten variables for sequence data.
    #     Input shapes: 
    #         upper_data: [time, var, pressure, height, width]
    #         surface_data: [time, var, height, width]
    #     Output shape: 
    #         [time, combined_var, height, width]
    #     """
    #     concat_arr, _ = pack(
    #         [rearrange(upper_data, "t v p h w -> t (v p) h w"), surface_data],
    #         "t * h w"
    #     )
    #     return concat_arr

    # def flatten_var(self, upper_data, surface_data):
    #     """
    #     Flatten variables for single timestep data.
    #     Input shapes: 
    #         upper_data: [var, pressure, height, width]
    #         surface_data: [var, height, width]
    #     Output shape: 
    #         [combined_var, height, width]
    #     """
    #     concat_arr, _ = pack(
    #         [rearrange(upper_data, "v p h w -> (v p) h w"), surface_data],
    #         "* h w"
    #     )
    #     return concat_arr

    # def flatten_stat(self, upper_mean, upper_std, surface_mean, surface_std):
    #     concat_mean, _ = pack(
    #         [rearrange(upper_mean, "v p h w -> (v p) h w"), surface_mean],
    #         "* h w"
    #     )
    #     concat_std, _ = pack(
    #         [rearrange(upper_std, "v p h w -> (v p) h w"), surface_std],
    #         "* h w"
    #     )

    #     return concat_mean, concat_std       

    def map_var_name_for_Aurora(self, var_name: str) -> str:
        """
        Map variable names to Aurora's expected names.
        """
        var_name_mapping = {
            "t2m": "2t",
            "u10": "10u",
            "v10": "10v",
            "msl": "msl",
        }
        if var_name in var_name_mapping:
            return var_name_mapping[var_name]
        else:
            return var_name

    def get_latitude_longitude(self):
        """
        Get latitude and longitude ranges from start_date_hour.
        """
        upper_path, sfc_path = self._dt_to_path(self.start_date_hour)
        upper_nc = xr.open_dataset(upper_path).load()
        latitude, longitude = \
            upper_nc.latitude.sel(latitude = slice(*self.latitude)).values, \
            upper_nc.longitude.sel(longitude = slice(*self.longitude)).values
        upper_nc.close()
        return torch.Tensor(latitude), torch.Tensor(longitude)
    
    def get_levels(self):
        """
        Get pressure levels from start_date_hour.
        """
        upper_path, _ = self._dt_to_path(self.start_date_hour)
        upper_nc = xr.open_dataset(upper_path).load()
        levels = upper_nc.pressure_level.values
        upper_nc.close()
        # print(f"{type(levels)=}, {levels=}")
        # return torch.Tensor(levels)
        # return tuple(levels.tolist())
        return tuple(levels)
    
    def get_static_vars_ds(self):
        _ds = xr.open_dataset(self.data_root_dir + "/static/static_vars.nc").load()
        _d = {
            "static_vars": {
                v: torch.Tensor(
                    _ds[v].sel(
                        latitude = slice(*self.latitude), longitude = slice(*self.longitude)
                    ).values
                ).squeeze() for v in self.static_variables
            }
        }
        return _d

    def _dt_to_path(self, date_hour: pd.Timestamp) -> str:
        dir_path = Path(self.data_root_dir) / date_hour.strftime(r"%Y/%Y%m/%Y%m%d")
        name = date_hour.strftime(r"%Y%m%d%H")

        return str(dir_path / f"{name}_upper.nc"), str(dir_path / f"{name}_sfc.nc")

    def __len__(self) -> int:
        # Account for sequence length and lead time in calculating dataset length
        duration = self.end_date_hour - self.start_date_hour - pd.Timedelta(hours = self.lead_time + self.rollout_step)
        return round(duration.total_seconds()) // (60 * 60)

    def _nc_to_dict(self, upper_nc, sfc_nc) -> dict:
        """
        Make the nc file can be used to form a single sample.
        """

        # print(f"{upper_nc=}")
        # print(f"{sfc_nc=}")

        _d = {
            "surf_vars": {
                self.map_var_name_for_Aurora(v): torch.Tensor(
                    sfc_nc[v].sel(
                        latitude = slice(*self.latitude),
                        longitude = slice(*self.longitude)
                    ).values,
                ).squeeze() for v in self.surface_variables
            },
            "atmos_vars": {
                v: torch.Tensor(
                    upper_nc[v].sel(
                        pressure_level = self.levels,
                        latitude = slice(*self.latitude),
                        longitude = slice(*self.longitude),
                    ).values
                ).squeeze() for v in self.upper_variables
            },
        }

        # print(
        #     upper_nc["u"].sel(
        #         pressure_level = self.levels,
        #         latitude = slice(*self.latitude),
        #         longitude = slice(*self.longitude)
        #     )
        # )

        return _d
    
    def concat_ts(self, list_of_ts: list[dict]):
        """
        To follow the format of Aurora input (i.e. [b, t, h, w] for each variable),
        we concatenate each varible for each dict element (surf_var and atoms_var).
        """
        stacked_dict = {
            'surf_vars': {
                var: torch.stack([d['surf_vars'][var] for d in list_of_ts], dim = 0)
                for var in list_of_ts[0]['surf_vars']
            },
            'atmos_vars': {
                var: torch.stack([d['atmos_vars'][var] for d in list_of_ts], dim = 0)
                for var in list_of_ts[0]['atmos_vars']
            }
        }
        return stacked_dict
        
    def __getitem__(self, index: int) -> tuple:
        """
        Implement the interface for Aurora.
        By definition, Aurora's input will be 2 timestamp, and output will be $(rollout_steps) timestamps.
        
        """
        # Build sequence of input times and output_times.
        # Input is start from 1th timestamp instead of 0.
        date_hour_inputs = [
            self.start_date_hour + pd.Timedelta(hours = index + i) \
            for i in range(2)
        ]
        date_hour_outputs = [
            date_hour_inputs[-1] + pd.Timedelta(hours = self.lead_time + i) \
            for i in range(self.rollout_step)
        ]

        # Load input times to create a "single" timestamp features.
        in_t_list = []
        for in_t in date_hour_inputs:
            upper_path_in, sfc_path_in = self._dt_to_path(in_t)
            upper_nc_in = xr.open_dataset(upper_path_in).load()
            sfc_nc_in = xr.open_dataset(sfc_path_in).load()
            in_t_list.append(self._nc_to_dict(upper_nc_in, sfc_nc_in))
            upper_nc_in.close()
            sfc_nc_in.close()
        # input_data = torch.stack( in_t_list, dim = 0 )
        input_data = self.concat_ts(in_t_list)

        out_t_list = []
        for out_t in date_hour_outputs:
            upper_path_out, sfc_path_out = self._dt_to_path(out_t)
            upper_nc_out = xr.open_dataset(upper_path_out).load()
            sfc_nc_out = xr.open_dataset(sfc_path_out).load()
            out_t_list.append(self._nc_to_dict(upper_nc_out, sfc_nc_out))
            upper_nc_out.close()
            sfc_nc_out.close()
        # output_data = torch.stack( out_t_list, dim = 0 )
        output_data = self.concat_ts(out_t_list)

        # # 2. Load each timestep's files, stack
        # upper_seq, surface_seq = [], []
        # for date_hour in date_hour_inputs:
        #     upper_path, surface_path = self._dt_to_path(date_hour)
        #     upper_nc = xr.open_dataset(upper_path).load()
        #     surface_nc = xr.open_dataset(surface_path).load()
        #     upper_data, surface_data = self._stack_nc(upper_nc, surface_nc)
        #     upper_nc.close()
        #     surface_nc.close()
        #     upper_seq.append(upper_data)
        #     surface_seq.append(surface_data)
        # # Stack into shape [seq_len, ...]
        # upper_seq = torch.stack(upper_seq, dim=0)
        # surface_seq = torch.stack(surface_seq, dim=0)
        # stacked_input = (upper_seq, surface_seq)

        # 3. Load target file
        # date_hour_target = date_hour_inputs[-1] + pd.Timedelta(hours=self.lead_time)
        # upper_path_target, surface_path_target = self._dt_to_path(date_hour_target)
        # upper_nc_target = xr.open_dataset(upper_path_target).load()
        # surface_nc_target = xr.open_dataset(surface_path_target).load()
        # stacked_target = self._stack_nc(upper_nc_target, surface_nc_target)
        # upper_nc_target.close()
        # surface_nc_target.close()

        # 4. Standardize if needed
        # if self.standardization:
        #     stacked_input = (
        #         self.standardize(stacked_input[0], self.upper_mean, self.upper_std),
        #         self.standardize(stacked_input[1], self.surface_mean, self.surface_std),
        #     )
        #     stacked_target = (
        #         self.standardize(stacked_target[0], self.upper_mean, self.upper_std),
        #         self.standardize(stacked_target[1], self.surface_mean, self.surface_std),
        #     )

        # 5. Flatten if needed
        # if self.flatten:
        #     data_input = self.flatten_var_seq(stacked_input[0], stacked_input[1])   # [seq_len, ...]
        #     data_target = self.flatten_var(stacked_target[0], stacked_target[1])    # [...]
        #     stat = (
        #         self.flatten_stat(
        #             self.upper_mean, self.upper_std,
        #             self.surface_mean, self.surface_std
        #         ) if self.get_stat else None
        #     )
        # else:
        #     # Alias for clarity.
        #     data_input = stacked_input
        #     data_target = stacked_target
        #     stat = (
        #         (self.upper_mean, self.upper_std, self.surface_mean, self.surface_std)
        #         if self.get_stat else None
        #     )

        # result = [data_input, data_target]

        # 6. Return non-flattened
        # if self.get_stat:
        #     result.append(stat)
        
        result = [ input_data, output_data ]

        if self.get_datetime:
            result.append(date_hour_inputs[-1].strftime("%Y-%m-%d %H:%M:%S"))
        
        return tuple(result)
