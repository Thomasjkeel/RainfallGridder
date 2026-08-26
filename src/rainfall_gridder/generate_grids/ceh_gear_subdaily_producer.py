import datetime

import numpy as np
import polars as pl
import scipy.interpolate
import xarray as xr
from scipy.spatial import cKDTree

from rainfall_gridder.generate_grids.alt_stat_diag_frac import (
    get_stat_disag_fraction_1h_grid,
    get_stat_disag_fraction_15min_grid,
)
from rainfall_gridder.utils.spatial_utils import calculate_gauge_to_grid_centre_distance

MAX_DISTANCE_TO_GAUGE_M = 50000


class CEHGEARSubDailyProducer:
    def __init__(
        self,
        rainfall_data: pl.DataFrame,
        rainfall_metadata: pl.DataFrame,
        station_id_col: str,
        time_step: datetime.datetime,
        time_res: str,
        precipitation_col: str,
        easting_col: str,
        northing_col: str,
        date_time_col: str,
        hour_at_start_of_day: int,
        verbose: bool,
    ):
        """
        CEH-GEAR subdaily producer.

        Parameters
        ----------

        """
        assert time_res in ["1h", "15m"], f"Data resolution needs to be either '15m' or '1h', currently: {time_res}."
        self.rainfall_metadata = rainfall_metadata
        self.time_step = time_step
        self.time_res = time_res
        self.precipitation_col = precipitation_col
        self.easting_col = easting_col
        self.northing_col = northing_col
        self.station_id_col = station_id_col
        self.date_time_col = date_time_col
        self.hour_at_start_of_day = hour_at_start_of_day
        self.verbose = verbose
        self.one_day_rainfall_data = self._get_one_day_rainfall_data(rainfall_data)
        self.one_day_daily_totals = self._get_daily_gauge_totals()
        self.gauge_daily_info = self._get_daily_info()
        self.gauge_daily_totals = self.gauge_daily_info[self.precipitation_col].to_numpy()

    def _get_one_day_rainfall_data(
        self,
        rainfall_data: pl.DataFrame,
    ) -> pl.DataFrame:
        # TODO: what if time-step is not at the start of the day?
        start = datetime.datetime(
            self.time_step.year,
            self.time_step.month,
            self.time_step.day,
            self.hour_at_start_of_day,
            0,  # TODO: check in minute effects things
            0,
        )
        end = start + datetime.timedelta(hours=24)  # e.g. 9:00 AM next day
        rainfall_data = rainfall_data.sort((self.station_id_col, self.date_time_col))
        return rainfall_data.filter(
            (pl.col(self.station_id_col).is_in(self.rainfall_metadata[self.station_id_col].unique().to_list()))
            & (pl.col(self.date_time_col) >= start)
            & (pl.col(self.date_time_col) < end)
        )

    def _get_daily_gauge_totals(self) -> pl.DataFrame:
        n_time_steps = 23 if self.time_res == "1h" else 95
        return (
            self.one_day_rainfall_data.group_by(self.station_id_col)
            .agg(
                [
                    pl.col(self.precipitation_col).sum().alias(self.precipitation_col),
                    pl.col(self.precipitation_col).count().alias("_timestep_count"),
                ]
            )
            .filter(pl.col("_timestep_count") >= n_time_steps)
            .drop("_timestep_count")
        )

    def _get_daily_info(self) -> pl.DataFrame:
        gauge_daily_info = self.one_day_daily_totals.join(
            self.rainfall_metadata.select([self.station_id_col, self.easting_col, self.northing_col]),
            on=self.station_id_col,
            how="inner",
        )
        gauge_daily_info = gauge_daily_info.with_columns(
            pl.struct([pl.col(self.easting_col), pl.col(self.northing_col)]).alias("points")
        )
        # Drop all daily totals that are NaN from distance calculation
        gauge_daily_info = gauge_daily_info.drop_nans(subset=[self.precipitation_col])

        # Important in current method that site_id is sorted as we are converting this data to numpy arrays
        return gauge_daily_info.sort(self.station_id_col)

    def _build_interpolators(
        self,
    ) -> tuple[
        scipy.interpolate.NearestNDInterpolator,
        scipy.interpolate.NearestNDInterpolator,
        scipy.interpolate.NearestNDInterpolator,
    ]:
        # Get coordinates as numpy arrays for interpolation
        gauge_eastings = self.gauge_daily_info[self.easting_col].to_numpy()
        gauge_northing = self.gauge_daily_info[self.northing_col].to_numpy()
        gauge_points = self.gauge_daily_info["points"].to_numpy()

        gauge_x_interpolator = scipy.interpolate.NearestNDInterpolator(gauge_points, gauge_eastings)
        gauge_y_interpolator = scipy.interpolate.NearestNDInterpolator(gauge_points, gauge_northing)
        daily_totals_interpolator = scipy.interpolate.NearestNDInterpolator(gauge_points, self.gauge_daily_totals)
        return gauge_x_interpolator, gauge_y_interpolator, daily_totals_interpolator

    def run_interpolation(
        self,
        x_coords: xr.DataArray,
        y_coords: xr.DataArray,
        x_grid: xr.DataArray,
        y_grid: xr.DataArray,
    ):
        gauge_x_interpolator, gauge_y_interpolator, daily_totals_interpolator = self._build_interpolators()
        gauge_x_grid = interpolate_values_onto_coordinate_grid(gauge_x_interpolator, x_grid, y_grid, x_coords, y_coords)
        gauge_y_grid = interpolate_values_onto_coordinate_grid(gauge_y_interpolator, x_grid, y_grid, x_coords, y_coords)
        daily_totals_grid = interpolate_values_onto_coordinate_grid(
            daily_totals_interpolator, x_grid, y_grid, x_coords, y_coords
        )
        return gauge_x_grid, gauge_y_grid, daily_totals_grid

    def calculate_distance_grid(
        self,
        land_mask: xr.DataArray,
        gauge_x_grid: xr.DataArray = None,
        gauge_y_grid: xr.DataArray = None,
    ) -> xr.DataArray:
        # TODO: put in another class
        x_coords, y_coords, x_grid, y_grid = get_xy_coordinate_grids(land_mask, return_coords=True)
        if not isinstance(gauge_x_grid, xr.DataArray) or not isinstance(gauge_y_grid, xr.DataArray):
            gauge_x_grid, gauge_y_grid, _ = self.run_interpolation(x_coords, y_coords, x_grid, y_grid)
        distance_grid = calculate_gauge_to_grid_centre_distance(x_grid, y_grid, gauge_x_grid, gauge_y_grid)
        distance_grid = np.round(distance_grid / 1000, 2)  # convert from metres to kilometres
        # mask out oceans
        distance_grid = distance_grid.where(land_mask)
        distance_grid["time"] = self.time_step
        return distance_grid

    def get_cells_to_stat_disag(
        self,
        land_mask: xr.DataArray,
        daily_totals_grid: xr.DataArray,
        distance_grid: xr.DataArray = None,
        max_distance_to_gauge_m: int = MAX_DISTANCE_TO_GAUGE_M,
    ) -> xr.DataArray:
        # TODO: put this method into another class
        # Different ways of doing this too
        if distance_grid is None:
            distance_grid = self.calculate_distance_grid(land_mask)
        # Get mask of cells for stat disaggregation (gap filling)
        daily_totals_grid_masked = daily_totals_grid.where(land_mask)
        daily_totals_grid_masked = daily_totals_grid_masked.where(daily_totals_grid != 0)
        daily_totals_grid_masked = daily_totals_grid_masked.where(distance_grid < max_distance_to_gauge_m)

        # TODO: check the part where I remove max distance is correct
        cells_to_stat_disag = (daily_totals_grid_masked.isnull() == land_mask).where(
            distance_grid < max_distance_to_gauge_m, 0
        )
        return cells_to_stat_disag

    def get_subdaily_rainfall_factors(
        self,
        land_mask: xr.DataArray,
        daily_totals_grid: xr.DataArray,
        one_day_gridded_daily: xr.Dataset,
        cells_to_stat_disag: xr.DataArray,
        gridded_rainfall_col: str,
    ) -> xr.Dataset:
        # Have some assert to check there are not too many time steps
        # TODO: check that this will always be the same order
        # 0. Get individual gauge coords for the day
        gauge_points = self.gauge_daily_info["points"].to_numpy()
        x_coords, y_coords, x_grid, y_grid = get_xy_coordinate_grids(land_mask, return_coords=True)

        grid_disag_func = (
            get_stat_disag_fraction_15min_grid if self.time_res == "15m" else get_stat_disag_fraction_1h_grid
        )

        # 1. Precompute nearest gauge for every grid cell
        grid_points = np.column_stack((x_grid.to_numpy().ravel(), y_grid.to_numpy().ravel()))

        tree = scipy.spatial.cKDTree(gauge_points)
        _, nearest_gauge_idx = tree.query(grid_points)
        nearest_gauge_idx = nearest_gauge_idx.reshape(x_grid.shape)

        # Look for any cells to stat disag
        masked_one_day_gridded_daily = one_day_gridded_daily[gridded_rainfall_col].where(cells_to_stat_disag)
        no_cells_to_disagg = bool(masked_one_day_gridded_daily.isnull().all())

        # 2. Calculate subdaily factor grid
        # 2.1 Format data before partioning and looping through. prefilter out gauge stations not in the day
        station_ids_in_day = self.gauge_daily_info[self.station_id_col].to_list()
        one_day_rainfall_data = self.one_day_rainfall_data.filter(pl.col(self.station_id_col).is_in(station_ids_in_day))
        one_day_rainfall_data.sort((self.station_id_col, self.date_time_col))

        # 3. regularilirse the data so no time steps missing in
        all_time_steps_gauge_data = (
            one_day_rainfall_data.select(self.date_time_col)
            .unique()
            .join(self.gauge_daily_info.select("station_id").unique(maintain_order=True), how="cross")
            .join(
                one_day_rainfall_data,
                on=[self.date_time_col, "station_id"],
                how="left",
            )
            .sort([self.date_time_col, "station_id"])
        )

        if len(all_time_steps_gauge_data) != len(one_day_rainfall_data):
            print(
                f"Warning: Data has been regularised so all time steps in day for all station ids. "
                f"Number of data points in day: before: {len(all_time_steps_gauge_data)}, after: {len(one_day_rainfall_data)}"
            )
        # 3.1 Partition pl.Dataframe into individual time steps
        all_time_steps_gauge_data_groups = all_time_steps_gauge_data.partition_by(self.date_time_col, as_dict=True)

        # 4. Loop through all time steps in day and compute factor grids
        all_subdaily_factor_grid = []

        for time_step, gauge_one_timestep in all_time_steps_gauge_data_groups.items():
            time_step = time_step[0]  # returned as a tuple, so need to get first item
            assert len(gauge_one_timestep) == gauge_points.shape[0], (
                f"The number of gauges with data ({len(gauge_one_timestep)}) need to be the same as number of gauges ({gauge_points.shape[0]})"
            )

            gauge_one_timestep_rainfall = gauge_one_timestep[self.precipitation_col].to_numpy()

            # Fast nearest-neighbour interpolation using precomputed
            # nearest-gauge indices.
            timestep_grid = gauge_one_timestep_rainfall[nearest_gauge_idx]

            timestep_grid = xr.DataArray(
                timestep_grid,
                coords=land_mask.coords,
                dims=land_mask.dims,
            )

            factor_grid = (timestep_grid / daily_totals_grid).where(land_mask)
            # Important to do before stat disagg
            factor_grid = factor_grid.fillna(0.0)

            if not no_cells_to_disagg:
                time_step_w_offset = time_step - datetime.timedelta(hours=self.hour_at_start_of_day)
                cells_to_stat_disag_frac = grid_disag_func(
                    masked_one_day_gridded_daily,
                    time_step_w_offset,
                )
                combined_factor_grid = factor_grid.where(cells_to_stat_disag_frac.isnull(), cells_to_stat_disag_frac)
            else:
                # There are no cells to stat disaggregate
                if self.verbose:
                    print("To remove: there are no cells to stat disagg", flush=True)
                combined_factor_grid = factor_grid

            # set time
            combined_factor_grid = combined_factor_grid.expand_dims(time=[time_step])
            all_subdaily_factor_grid.append(combined_factor_grid)
        all_subdaily_factor_grid_ds = xr.concat(all_subdaily_factor_grid, dim="time")
        return all_subdaily_factor_grid_ds

    def produce_ceh_gear(
        self,
        land_mask: xr.DataArray,
        one_day_gridded_daily: xr.Dataset,
        gridded_rainfall_col: str,
        output_rainfall_name: str = "rainfall",
    ) -> xr.Dataset:
        # 1. Get coord grids
        x_coords, y_coords, x_grid, y_grid = get_xy_coordinate_grids(land_mask, return_coords=True)
        # 2. Run interpolation for distance grid and daily totals
        gauge_x_grid, gauge_y_grid, daily_totals_grid = self.run_interpolation(x_coords, y_coords, x_grid, y_grid)
        distance_grid = self.calculate_distance_grid(land_mask, gauge_x_grid, gauge_y_grid)
        distance_grid["time"] = self.time_step

        # 3. Get cells to statistically disaggregate based on interpolated daily totals
        cells_to_stat_disag = self.get_cells_to_stat_disag(land_mask, daily_totals_grid)
        cells_to_stat_disag["time"] = self.time_step
        one_day_gridded_daily_masked = one_day_gridded_daily.where(land_mask)

        # 4. Compute subdaily rainfall factors
        all_factor_grid_ds = self.get_subdaily_rainfall_factors(
            land_mask,
            daily_totals_grid,
            one_day_gridded_daily_masked,
            cells_to_stat_disag,
            gridded_rainfall_col,
        )
        # 5. Downscale gridded daily by subdaily factors
        ceh_gear_one_day = one_day_gridded_daily[gridded_rainfall_col] * all_factor_grid_ds

        # 6. Convert to dataset
        ceh_gear_one_day = ceh_gear_one_day.to_dataset(name=output_rainfall_name)
        ceh_gear_one_day["min_dist_km"] = distance_grid
        ceh_gear_one_day["stat_disag"] = cells_to_stat_disag
        return ceh_gear_one_day


def interpolate_values_onto_coordinate_grid(
    interpolator: scipy.interpolate.NearestNDInterpolator,
    x_grid: xr.DataArray,
    y_grid: xr.DataArray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> xr.DataArray:
    return xr.DataArray(
        interpolator(x_grid.values, y_grid.values),
        dims=["y", "x"],
        coords={"x": x_coords, "y": y_coords},
    )


def get_xy_coordinate_grids(ds: xr.Dataset, return_coords: bool):
    x_coords = ds.x.values
    y_coords = ds.y.values
    y_grid, x_grid = xr.broadcast(ds.y, ds.x)
    if return_coords:
        return x_coords, y_coords, x_grid, y_grid
    return x_grid, y_grid
