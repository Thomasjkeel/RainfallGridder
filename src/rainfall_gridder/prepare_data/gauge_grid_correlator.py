from pathlib import Path

import polars as pl
import scipy.stats
import xarray as xr

from rainfall_gridder.prepare_data.data_combiner import GaugeVsGriddedRainfallMatcher
from rainfall_gridder.utils import spatial_utils, xarray_utils


class GaugeVsGriddedCorrelator:
    def __init__(
        self,
        gauge_data: pl.DataFrame,
        gauge_metadata: pl.DataFrame,
        nearest_gridded_daily: xr.Dataset,
        station_id: str,
        precipitation_col: str,
        gridded_rainfall_col: str,
        date_time_col: str,
        start_date_col: str,
        end_date_col: str,
        station_id_col: str,
        easting_col: str,
        northing_col: str,
        rainfall_offset_hours: int,
        aggregate_gauge_to_daily: bool = True,
    ):
        """
        Will correlate Rain gauge data with nearest gridded data.

        TODO: make sure the combining of gauge name is done in order i.e. 1-2 not 2-1

        Parameters
        ----------
        gauge_data:
            Rainfall gauge data
        gauge_metadata:
            Details of rain gauge data
        """
        # filter to the single station ID
        self.gauge_data = gauge_data.filter(pl.col(station_id_col) == station_id).sort(by=date_time_col)
        self.gauge_metadata = gauge_metadata.filter(pl.col(station_id_col) == station_id)
        self.station_id = station_id
        self.precipitation_col = precipitation_col
        self.gridded_rainfall_col = gridded_rainfall_col
        self.date_time_col = date_time_col
        self.start_date_col = start_date_col
        self.end_date_col = end_date_col
        self.station_id_col = station_id_col
        self.easting_col = easting_col
        self.northing_col = northing_col
        self.rainfall_offset_hours = rainfall_offset_hours

        self.nearest_gridded_daily = self._load_daily_nearest_gridded_daily(nearest_gridded_daily)
        self.nearest_gridded_daily.load()

        if aggregate_gauge_to_daily:
            self.gauge_data = self._aggregate_gauge_subdaily_to_daily()
        self.combined_data = self._join_gauge_to_grid()

    def _aggregate_gauge_subdaily_to_daily(self) -> pl.DataFrame:
        return (
            self.gauge_data.drop_nulls()
            .group_by_dynamic(
                self.date_time_col,
                every="1d",
                offset=f"{self.rainfall_offset_hours}h",
                label="left",
            )
            .agg(pl.col(self.precipitation_col).sum())
        )

    def _load_daily_nearest_gridded_daily(self, nearest_gridded_daily: xr.Dataset) -> xr.Dataset:
        """
        Load daily gridded data.

        TODO: This method need some work, because what if the day doesn't start at 00:00.
        TODO: also need to assert the data is actually daily
        """
        nearest_gridded_daily = spatial_utils.get_nearest_grid_cell(
            nearest_gridded_daily,
            easting=self.gauge_metadata[self.easting_col][0],
            northing=self.gauge_metadata[self.northing_col][0],
        )

        nearest_gridded_daily = xarray_utils.replace_daily_time_step_hour_with_zero(
            nearest_gridded_daily, time_col="time"
        )

        return self._subset_gridded_data_to_start_and_end_of_gauge(nearest_gridded_daily)

    def _join_gauge_to_grid(self):
        s_date = self.gauge_metadata[self.start_date_col][0]
        e_date = self.gauge_metadata[self.end_date_col][0]
        gauge_gridded_matcher = GaugeVsGriddedRainfallMatcher(
            [self.precipitation_col],
            output_col_name="",
            rainfall_offset_hours=self.rainfall_offset_hours,
            date_time_col=self.date_time_col,
        )
        nearest_gridded_daily_cell_df = gauge_gridded_matcher.prepare_gridded_daily(
            self.nearest_gridded_daily,
            s_date=s_date,
            e_date=e_date,
            rain_col=self.gridded_rainfall_col,
        )
        combined_gauge_gridded = gauge_gridded_matcher.join_daily_gauge_and_gridded(
            self.gauge_data, nearest_gridded_daily_cell_df
        )
        return combined_gauge_gridded

    def _subset_gridded_data_to_start_and_end_of_gauge(self, nearest_gridded_daily: xr.Dataset) -> xr.Dataset:
        """
        Clip gridded data so only extends between start and end date of gauge data.

        Parameters
        ----------
        nearest_gridded_daily:
            Nearest grid cell of daily rainfall

        Returns
        -------
        nearest_gridded_daily_clipped:
            Nearest grid cell of daily rainfall clipped to start and end datetime of gauge

        Raises
        ------
        ValueError:
            If there is no overlap between gauge data and gridded daily rainfall

        """
        start_datetime = self.gauge_metadata[self.start_date_col][0]
        end_datetime = self.gauge_metadata[self.end_date_col][0]
        nearest_gridded_daily_clipped = nearest_gridded_daily.sel(time=slice(start_datetime, end_datetime))
        if nearest_gridded_daily_clipped["time"].size == 0:
            raise ValueError(
                "No overlap between the daily gridded data and the inputted gauge data. "
                f"Gauge data runs from {start_datetime} to {end_datetime}, "
                f"whereas the gridded data runs from "
                f"{nearest_gridded_daily['time'].min().data} "
                f"to {nearest_gridded_daily['time'].max().data}."
            )
        return nearest_gridded_daily_clipped

    def get_corr(self):
        r_result = scipy.stats.pearsonr(
            self.combined_data[self.precipitation_col],
            self.combined_data[self.gridded_rainfall_col],
        ).statistic
        rho_result = scipy.stats.spearmanr(
            self.combined_data[self.precipitation_col],
            self.combined_data[self.gridded_rainfall_col],
        ).statistic
        return r_result, rho_result


class BatchGaugeVsGriddedCorrelator(GaugeVsGriddedCorrelator):
    """
    Run the correlator on a list of station IDs and return and/or save the prepared data.
    """

    def __init__(
        self,
        gauge_data: pl.DataFrame,
        gauge_metadata: pl.DataFrame,
        gridded_rainfall_data: xr.Dataset,
        gridded_rainfall_col: str,
        station_ids_to_correlate: list,
        station_id_col: str,
        precipitation_col: str,
        date_time_col: str,
        start_date_col: str,
        end_date_col: str,
        easting_col: str,
        northing_col: str,
        rainfall_offset_hours: int,
        output_dir: Path,
        verbose: bool,
        correlation_threshold: float,
        aggregate_gauge_to_daily: bool = True,
    ):
        """
        Will correlation for multiple station comparing rain gauge data with gridded data nearest to it.

        TODO: make sure the combining of gauge name is done in order i.e. 1-2 not 2-1

        Parameters
        ----------
        gauge_data:
            Rainfall gauge data
        gauge_metadata:
            Details of rain gauge data
        station_ids_to_correlate:
            List of ids to run through the correlator.
        output_dir:
            Output directory for data files
        verbose:
            Whether to print progress as algorithm is run
        correlation_threshold:
            rain gauges lower than this will be flagged for removal
        aggregate_gauge_to_daily:
            Whether to aggregated rain gauge to daily time res (deafult: True)
        Returns
        -------
        """
        self.gauge_data = gauge_data
        self.gauge_metadata = gauge_metadata
        self.station_ids_to_correlate = station_ids_to_correlate
        self.gridded_rainfall_data = gridded_rainfall_data
        self.station_id_col = station_id_col
        self.precipitation_col = precipitation_col
        self.gridded_rainfall_col = gridded_rainfall_col
        self.date_time_col = date_time_col
        self.start_date_col = start_date_col
        self.end_date_col = end_date_col
        self.easting_col = easting_col
        self.northing_col = northing_col
        self.rainfall_offset_hours = rainfall_offset_hours
        self.correlation_threshold = correlation_threshold
        self.output_dir = output_dir
        self.verbose = verbose
        self.aggregate_gauge_to_daily = aggregate_gauge_to_daily

    def run_correlator(self):
        station_ids_to_remove = []
        for station_id in self.station_ids_to_correlate:
            # Subset the metadata, data
            metadata_one_station = self.gauge_metadata.filter(pl.col(self.station_id_col) == station_id)
            if metadata_one_station.is_empty():
                if self.verbose:
                    print(f"Station ID: {station_id} is not included in the metadata", flush=True)
                    continue
            data_one_station = self.gauge_data.filter(
                pl.col(self.station_id_col).is_in(metadata_one_station[self.station_id_col].unique().to_list())
            )
            if data_one_station.is_empty():
                if self.verbose:
                    print(f"Station ID: {station_id} is not included in the data", flush=True)
                    continue

            try:
                gauge_grid_correlator = GaugeVsGriddedCorrelator(
                    gauge_data=data_one_station,
                    gauge_metadata=metadata_one_station,
                    nearest_gridded_daily=self.gridded_rainfall_data,
                    station_id=station_id,
                    precipitation_col=self.precipitation_col,
                    gridded_rainfall_col=self.gridded_rainfall_col,
                    date_time_col=self.date_time_col,
                    start_date_col=self.start_date_col,
                    end_date_col=self.end_date_col,
                    station_id_col=self.station_id_col,
                    easting_col=self.easting_col,
                    northing_col=self.northing_col,
                    rainfall_offset_hours=self.rainfall_offset_hours,
                    aggregate_gauge_to_daily=self.aggregate_gauge_to_daily,
                )
            except ValueError as ve:
                station_ids_to_remove.append(station_id)
                if self.verbose:
                    print(station_id, ve, flush=True)
                    print(station_id, "flagged for removal", flush=True)
                continue

            try:
                r_result, rho_result = gauge_grid_correlator.get_corr()
            except ValueError as ve:
                station_ids_to_remove.append(station_id)
                if self.verbose:
                    print(station_id, "failed, probably all NaN", ve, flush=True)
                    print(station_id, "flagged for removal", flush=True)
                continue
            if self.verbose:
                print(station_id, r_result, rho_result, flush=True)
            if r_result > self.correlation_threshold or rho_result > self.correlation_threshold:
                pass
            else:
                station_ids_to_remove.append(station_id)
                if self.verbose:
                    print(station_id, "flagged for removal", flush=True)

        self.corrd_metadata = self.gauge_metadata.filter(~pl.col(self.station_id_col).is_in(station_ids_to_remove))

    @classmethod
    def run(cls, save_metadata: bool, return_metadata: bool, **kwargs) -> None | pl.DataFrame:
        """
        Run the correlator on a list of station IDs and return and/or save the prepared data.

        Parameters
        ----------
        save_metadata:
            Whether to save metadata to output directory
        return_metadata:
            Whether to return metadata dataframe

        Returns
        -------
        metadata:
            Metadata of data run through algorithm

        """
        batch_correlator = cls(**kwargs)
        if batch_correlator.verbose:
            print("Starting Gauge vs Gridded Correlator", flush=True)
        batch_correlator.run_correlator()
        if save_metadata:
            if batch_correlator.verbose:
                print(f"Saving data to {batch_correlator.output_dir}", flush=True)
            batch_correlator.save_corrd_metadata()
        else:
            if batch_correlator.verbose:
                print("Data not saved", flush=True)
        if return_metadata:
            return batch_correlator.corrd_metadata

    def save_corrd_metadata(self) -> None:
        if self.corrd_metadata is None:
            raise RuntimeError("You must call run_correlator() before save_corrd_metadata()")
        assert len(self.corrd_metadata.filter(pl.col("file_path").is_duplicated())) == 0, (
            "Problem with metadata as duplicate filepaths"
        )
        self.corrd_metadata.write_parquet(self.output_dir / "corrd_metadata.parquet")
        if self.verbose:
            print(
                "Gauge metadata filtered by correlation to nearest grid cell available "
                f"at: {self.output_dir / 'corrd_metadata.parquet'}",
                flush=True,
            )
