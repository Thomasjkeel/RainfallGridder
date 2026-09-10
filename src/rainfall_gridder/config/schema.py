import datetime
from pathlib import Path

import fsspec
import polars as pl
import xarray as xr
import zarr
from polars.exceptions import ComputeError, InvalidOperationError
from pydantic import BaseModel, Field, model_validator


class ColumnConfig(BaseModel):
    station_id_col: str = "station_id"
    station_name_col: str = "station_name"
    date_time_col: str = "date_time"
    start_date_col: str = "start_date"
    end_date_col: str = "end_date"
    easting_col: str = "easting"
    northing_col: str = "northing"
    precipitation_col: str = "precipitation"


class RainGaugeMetadataConfig(BaseModel):
    path: Path


class RainGaugeDataConfig(BaseModel):
    path: Path


class GriddedRainfallConfig(BaseModel):
    path: Path | list[Path]
    rename: dict[str, str] = Field(default_factory=dict)
    from_object_store: bool = False
    object_store_config: dict[str, str] = Field(default_factory=dict)


class WorkflowConfig(BaseModel):
    rainfall_data: RainGaugeDataConfig
    rainfall_metadata: RainGaugeMetadataConfig
    data_columns: ColumnConfig
    gridded_rainfall_data: GriddedRainfallConfig
    gridded_rainfall_col: str
    workflow_start_date: datetime.datetime | datetime.date
    workflow_end_date: datetime.datetime | datetime.date
    output_dir: Path
    rainfall_offset_hours: int
    n_hours: int
    verbose: bool = False
    input_crs: str
    time_res: str
    smallest_rainfall_amount: float
    min_n_neighbours: int
    qc_framework: str
    nearby_rainfall_data_loader_kwargs: dict
    correlation_threshold: float
    output_rainfall_name: str
    min_n_timesteps: int = 100
    batch_size: int = 5
    output_zarr_name: str = "final_gridded_data"

    @model_validator(mode="after")
    def preformat_workflow_datetimes(self):
        if type(self.workflow_start_date) is datetime.date:
            self.workflow_start_date = datetime.datetime.combine(self.workflow_start_date, datetime.time.min)

        if type(self.workflow_end_date) is datetime.date:
            self.workflow_end_date = datetime.datetime.combine(self.workflow_end_date, datetime.time(23, 59, 59))

        return self

    def load_rainfall_data(self) -> pl.DataFrame:
        """
        Loads the entire rainfall dataset and will look for it to be either:
        1. .parquet
        2. .csv
        3. a directory containing parquet or csv files
        """
        rainfall_data_path = Path(self.rainfall_data.path)
        time_subset = (pl.col(self.data_columns.date_time_col) >= self.workflow_start_date) & (
            pl.col(self.data_columns.date_time_col) <= self.workflow_end_date
        )
        if rainfall_data_path.suffix == ".parquet":
            return pl.read_parquet(rainfall_data_path, try_parse_hive_dates=True).filter(time_subset)

        if rainfall_data_path.suffix == ".csv":
            return pl.read_csv(rainfall_data_path, try_parse_dates=True).filter(time_subset)

        try:
            return pl.scan_parquet(rainfall_data_path, try_parse_hive_dates=True).filter(time_subset).collect()
        except (ComputeError, InvalidOperationError):
            try:
                return pl.scan_csv(rainfall_data_path, try_parse_dates=True).filter(time_subset).collect()
            except (ComputeError, InvalidOperationError) as err:
                raise ValueError(f"Problem with files in rainfall data input path: {rainfall_data_path}") from err

    def load_rainfall_metadata(self) -> pl.DataFrame:
        rainfall_metadata_path = Path(self.rainfall_metadata.path)

        if rainfall_metadata_path.suffix == ".parquet":
            return pl.read_parquet(rainfall_metadata_path)

        if rainfall_metadata_path.suffix == ".csv":
            return pl.read_csv(rainfall_metadata_path)

        raise ValueError(f"Rainfall metadata path needs to be '.csv' or '.parquet'. Path: {rainfall_metadata_path}")

    def load_gridded_rainfall(self) -> xr.Dataset:
        if self.gridded_rainfall_data.from_object_store:
            fdri_fs = fsspec.filesystem(
                "s3",
                asynchronous=True,
                anon=True,
                endpoint_url=self.gridded_rainfall_data.object_store_config["endpoint_url"],
            )
            data_zstore = zarr.storage.FsspecStore(fdri_fs, path=self.gridded_rainfall_data.object_store_config["path"])
            ds = xr.open_zarr(data_zstore, decode_times=True, decode_cf=True)
        else:
            if isinstance(self.gridded_rainfall_data.path, list):
                ds = xr.open_mfdataset(self.gridded_rainfall_data.path)
            else:
                ds = xr.open_dataset(self.gridded_rainfall_data.path)
        if self.gridded_rainfall_data.rename:
            ds = ds.rename(self.gridded_rainfall_data.rename)
        assert self.gridded_rainfall_col in ds.data_vars, f"{self.gridded_rainfall_col} not in gridded_rainfall_data"
        return ds.sel(time=slice(self.workflow_start_date, self.workflow_end_date))
