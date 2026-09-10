from pathlib import Path

import polars as pl
import xarray as xr

from rainfall_gridder.prepare_data import data_combiner, data_formatting, metadata_preparer
from rainfall_gridder.utils import spatial_utils, xarray_utils


class DataPreparer:
    """
    Main data preparing algorithm.
    """

    def __init__(
        self,
        rainfall_data: pl.DataFrame,
        rainfall_metadata: pl.DataFrame,
        station_id_col: str,
        station_name_col: str,
        precipitation_col: str,
        date_time_col: str,
        start_date_col: str,
        end_date_col: str,
        easting_col: str,
        northing_col: str,
        gridded_rainfall_data: xr.Dataset,
        gridded_rainfall_col: str,
        rainfall_offset_hours: int,
        output_dir: str | Path,
        min_n_timesteps: int,
        verbose: bool = False,
    ):
        """
        Main data preparer for gridded workflow.

        Parameters
        ----------
        rainfall_data:
            Rainfall gauge data
        rainfall_metadata:
            Details of rain gauge data
        gridded_rainfall_col:
            Name of rainfall variable in the gridded_rainfall_data
        rainfall_offset_hours:
            First hour of the rainfall day (e.g. 9 if running from 9am to 8.59am)
        output_dir:
            Output directory for data files
        min_n_timesteps:
            Minimum number of timesteps needed in rainfall_data to be considered valid
        verbose:
            Whether to print progress as algorithm is run (default: False)

        """
        self.station_id_col = station_id_col
        self.station_name_col = station_name_col
        self.precipitation_col = precipitation_col
        self.date_time_col = date_time_col
        self.start_date_col = start_date_col
        self.end_date_col = end_date_col
        self.easting_col = easting_col
        self.northing_col = northing_col
        self.rainfall_offset_hours = rainfall_offset_hours
        self.output_dir = output_dir
        self.min_n_timesteps = min_n_timesteps
        self.gridded_rainfall_col = gridded_rainfall_col
        self.verbose = verbose

        # Prepare data inputs
        self.rainfall_data = self._prepare_data(rainfall_data)
        self.rainfall_metadata = self._prepare_metadata(rainfall_metadata)
        self.gridded_rainfall_data = self._prepare_gridded_rainfall_data(gridded_rainfall_data)

        # empty final outputs
        self.prepared_data = None
        self.prepared_metadata = None

    def _remove_duplicates_in_metadata(self, metadata: pl.DataFrame) -> pl.DataFrame:
        return metadata.unique(
            subset=[self.station_id_col]
        )  # TODO: this would leave wrong coords if it returns first unique

    def _prepare_metadata(self, rainfall_metadata: pl.DataFrame) -> pl.DataFrame:
        rainfall_metadata = self._remove_duplicates_in_metadata(rainfall_metadata)
        try:
            metadata_preparer.add_completeness_to_metadata(
                self.rainfall_data,
                rainfall_metadata,
                station_id_col=self.station_id_col,
                date_time_col=self.date_time_col,
            )
        except ValueError as ve:
            print(ve)

        rainfall_metadata = metadata_preparer.add_completeness_to_metadata(
            self.rainfall_data, rainfall_metadata, station_id_col=self.station_id_col, date_time_col=self.date_time_col
        )
        rainfall_metadata = data_formatting.group_metadata_by_station_locations(
            rainfall_metadata, easting_col=self.easting_col, northing_col=self.northing_col
        )
        return data_formatting.add_blank_file_path_to_metadata(rainfall_metadata)

    def _prepare_gridded_rainfall_data(self, gridded_rainfall_data: xr.Dataset) -> xr.Dataset:
        for data_var in ["x", "y", "time", self.gridded_rainfall_col]:
            assert data_var in gridded_rainfall_data, (
                f"Expecting data variable: '{data_var}' in gridded rainfall data. "
                "Please add this variable, or rename its equivalent."
            )
        gridded_rainfall_data = xarray_utils.subset_gridded_data_to_metadata_bounds(
            gridded_rainfall_data, self.rainfall_metadata, self.easting_col, self.northing_col
        )
        return gridded_rainfall_data

    def _prepare_data(self, data: pl.DataFrame) -> pl.DataFrame:
        return data_formatting.set_negative_precip_values_to_none(data, precip_col=self.precipitation_col)

    @classmethod
    def run(
        cls, save_data: bool, return_data: bool, partition_by_columns: list = None, **kwargs
    ) -> None | tuple[pl.DataFrame, pl.DataFrame]:
        """
        Run the data preparer and return and/or save the prepared data.

        Parameters
        ----------
        save_data:
            Whether to save data to output directory
        return_data:
            Whether to return dataframes
        partition_by_columns:
            List of columns to partition the parquet files by if saving outputs

        Returns
        -------
        prepared_data:
            Data run through algorithm
        prepared_metadata:
            Metadata of data run through algorithm

        """
        data_preparer = cls(**kwargs)
        if data_preparer.verbose:
            print("Preparing data for gridder", flush=True)
        data_preparer.prepare_data_and_metadata_for_gridding()
        if save_data:
            if data_preparer.verbose:
                print(f"Saving data to {data_preparer.output_dir}", flush=True)
            data_preparer.save_prepared_data(partition_by_columns)
            data_preparer.save_prepared_metadata()
        else:
            if data_preparer.verbose:
                print("Data not saved", flush=True)
        if return_data:
            return data_preparer.prepared_data, data_preparer.prepared_metadata

    def prepare_data_and_metadata_for_gridding(self) -> None:
        prepared_data_list = []
        prepared_metadata_list = []

        # Loop through each station id group (group may be multiple gauges if a site has a backup)
        for station_group_id in self.rainfall_metadata["station_group_id"].unique():
            metadata_one_group = self.rainfall_metadata.filter(pl.col("station_group_id") == station_group_id)
            data_one_group = self.rainfall_data.filter(
                pl.col(self.station_id_col).is_in(metadata_one_group[self.station_id_col].unique().to_list())
            )

            # Sort and drop duplicates
            data_one_group = data_one_group.sort(self.date_time_col).unique()

            if len(data_one_group[self.station_id_col].unique()) > 1:
                print(f"Group ID: {station_group_id} has {len(data_one_group[self.station_id_col].unique())} members")
                # create pivot of data
                data_one_group_pivot = data_one_group.pivot(
                    values=self.precipitation_col, index=self.date_time_col, on=self.station_id_col
                ).sort(by=self.date_time_col)

                # if duplicate exist, merge segments
                gauge_combiner = data_combiner.RainGaugeSegmentCombiner(
                    pivoted_gauge_data=data_one_group_pivot,
                    metadata=metadata_one_group,
                    station_id_col=self.station_id_col,
                )
                nearest_daily_gridded_cell = spatial_utils.get_nearest_grid_cell(
                    self.gridded_rainfall_data,
                    easting=metadata_one_group[self.easting_col][0],
                    northing=metadata_one_group[self.northing_col][0],
                )

                nearest_daily_gridded_cell = xarray_utils.replace_daily_time_step_hour_with_zero(
                    nearest_daily_gridded_cell, time_col="time"
                )
                nearest_daily_gridded_cell.load()

                combined_data = gauge_combiner.loop_through_and_merge_data(
                    nearest_daily_gridded_cell,
                    date_time_col=self.date_time_col,
                    rain_col=self.gridded_rainfall_col,
                    rainfall_offset_hours=self.rainfall_offset_hours,
                )
                station_name = gauge_combiner.combined_station_col_name

                # unpivot data
                data_one_group = combined_data.unpivot(
                    index=[self.date_time_col],
                    on=[station_name],
                    variable_name=self.station_id_col,
                    value_name=self.precipitation_col,
                )

            else:
                station_name = metadata_one_group[self.station_id_col][0]

            # Save data if at least N months worth of non-null record
            if len(data_one_group.drop_nulls()) >= self.min_n_timesteps:
                if self.verbose:
                    print(f"Adding group ID: {station_group_id}", flush=True)
                output_file_name = str(
                    data_combiner.build_output_path(
                        base_dir=self.output_dir / "data", id_col_name=self.station_id_col, station_id=station_name
                    )
                )
                metadata_one_group = metadata_one_group.with_columns(pl.lit(output_file_name).alias("file_path"))
                data_one_group = data_one_group.select(
                    [self.date_time_col, self.precipitation_col, self.station_id_col]
                )
                prepared_data_list.append(data_one_group)
            else:
                if self.verbose:
                    print(
                        f"{station_name} being ignored as not more than {self.min_n_timesteps} time steps.", flush=True
                    )
            if len(metadata_one_group) > 1:
                if self.verbose:
                    print(f"merging metadata of {station_name}", flush=True)
                metadata_merger = metadata_preparer.MetadataMerger(
                    metadata=metadata_one_group,
                    cols_to_check_identical=[self.easting_col, self.northing_col, "station_group_id", "file_path"],
                    cols_to_combine=[col for col in [self.station_id_col, self.station_name_col] if col],
                    start_date_col=self.start_date_col,
                    end_date_col=self.end_date_col,
                )
                merged_metadata = metadata_merger.merge_group_metadata(
                    group_name=station_name,
                    group_name_col=self.station_id_col,
                    min_datetime=data_one_group[self.date_time_col].min(),
                    max_datetime=data_one_group[self.date_time_col].max(),
                )
                prepared_metadata_list.append(merged_metadata.select(self.rainfall_metadata.columns))
            else:
                prepared_metadata_list.append(metadata_one_group.select(self.rainfall_metadata.columns))
        self.prepared_data = pl.concat(prepared_data_list)
        self.prepared_metadata = pl.concat(prepared_metadata_list, how="diagonal_relaxed")

    def save_prepared_data(self, partition_by_columns: list = None) -> None:
        """
        Save data that has been prepared for gridding.

        Parameters
        ----------
        partition_by_columns:
            Columns that decide the partitioning of the output parquet file structure (default is station_id_col)
        """
        if partition_by_columns is None:
            partition_by_columns = [self.station_id_col]

        if self.prepared_data is None:
            raise RuntimeError("You must call prepare_data_and_metadata_for_gridding() before save_prepared_data()")

        assert len(self.prepared_metadata.filter(pl.col("file_path").is_duplicated())) == 0, (
            "Problem with metadata as duplicate filepaths"
        )
        # Save partitioned parquet file
        (
            self.prepared_data.sort(self.date_time_col).write_parquet(
                self.output_dir / "data",
                partition_by=partition_by_columns,
            )
        )
        if self.verbose:
            print(f"prepared gauge data available at: {self.output_dir / 'data/'}", flush=True)

    def save_prepared_metadata(self) -> None:
        if self.prepared_metadata is None:
            raise RuntimeError("You must call prepare_data_and_metadata_for_gridding() before save_prepared_metadata()")
        assert len(self.prepared_metadata.filter(pl.col("file_path").is_duplicated())) == 0, (
            "Problem with metadata as duplicate filepaths"
        )
        self.prepared_metadata.write_parquet(self.output_dir / "prepared_metadata.parquet")
        if self.verbose:
            print(f"prepared gauge metadata available at: {self.output_dir / 'prepared_metadata.parquet'}", flush=True)
