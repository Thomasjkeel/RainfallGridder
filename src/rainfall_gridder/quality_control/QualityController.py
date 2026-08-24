from pathlib import Path
import polars as pl
import rainfallqc
from rainfallqc.qc_frameworks.inbuilt_qc_frameworks import NON_ROWWISE_QC_CHECKS, NON_ROWWISE_QC_CONVERTER

from rainfall_gridder.quality_control.apply_intenseQC_rulebase import apply_intenseQC_rulebase
from rainfall_gridder.quality_control.nearby_rainfall_data_loader import NearbyRainfallDataLoader
from rainfall_gridder.utils import spatial_utils


time_res_to_n_time_steps_in_day = {"15m": 96, "1h": 24}


class QualityController:
    """
    Main quality control running algorithm.
    """

    def __init__(
        self,
        rainfall_data: pl.DataFrame,
        rainfall_metadata: pl.DataFrame,
        station_id_col: str,
        station_name_col: str,
        date_time_col: str,
        precipitation_col: str,
        easting_col: str,
        northing_col: str,
        start_date_col: str,
        end_date_col: str,
        input_crs: str,
        min_n_timesteps: int,
        output_dir: str | Path,
        time_res: str,
        smallest_rainfall_amount: int | float,
        min_n_neighbours: int,
        qc_framework: str,
        nearby_rainfall_data_loader_kwargs: dict = {},
        verbose: bool = False,
    ):
        """
        Quality control part of gridded workflow.

        Parameters
        ----------
        rainfall_data:
            Rainfall gauge data
        rainfall_metadata:
            Details of rain gauge data
        input_crs:
            Projection of the east_west and north_south cols of the input data
        output_dir:
            Output directory for data files
        min_n_timesteps:
            Minimum number of timesteps needed in rainfall_data to be considered valid
        time_res:
            Resolution of data (i.e. hourly or 15 min denoted: '1h' or '15m')
        smallest_rainfall_amount:
            Smallest measurable rainfall amount
        min_n_neighbours:
            Minimum number of nearby rain gauges allowed for neighbourhood QC checks.
        qc_framework:
            QC framework to run (see rainfallqc.qc_frameworks/inbuilt_qc_frameworks for options or build your own by looking at RainfallQC docs)
        nearby_rainfall_data_loader_kwargs:
            Any additional arguments to override the defaults of the nearby data loader i.e. distance_threshold and  n_closest (default is {})
        verbose:
            Whether to print progress as algorithm is run (default: False)
        """
        self.station_id_col = station_id_col
        self.station_name_col = station_name_col
        self.date_time_col = date_time_col
        self.precipitation_col = precipitation_col
        self.easting_col = easting_col
        self.northing_col = northing_col
        self.start_date_col = start_date_col
        self.end_date_col = end_date_col
        self.input_crs = self._validate_input_crs(input_crs)
        self.min_n_timesteps = min_n_timesteps
        self.time_res = self._validate_time_res(time_res)
        self.smallest_rainfall_amount = smallest_rainfall_amount
        self.min_n_neighbours = min_n_neighbours
        self.qc_framework = qc_framework
        self.nearby_rainfall_data_loader_kwargs = nearby_rainfall_data_loader_kwargs
        self.output_dir = output_dir
        self.verbose = verbose

        if self.qc_framework == "intenseqc_rulebase_only":
            self.qc_kwargs, self.qc_methods_to_run = self._set_up_intenseqc_framework()

        else:
            raise ValueError(
                f"QC framework: '{self.qc_framework}' not recognised, please select from: 'intenseqc_rulebase_only'"
            )

        if "latitude" not in rainfall_metadata.columns or "longitude" not in rainfall_metadata.columns:
            self.rainfall_metadata = self._add_latlon_to_rainfall_metadata(rainfall_metadata)
        else:
            self.rainfall_metadata = rainfall_metadata
        self.rainfall_data = rainfall_data

    def _add_latlon_to_rainfall_metadata(self, rainfall_metadata: pl.DataFrame) -> pl.DataFrame:
        return spatial_utils.crs_to_crs(
            rainfall_metadata,
            crs_in=self.input_crs,
            crs_out="EPSG:4326",
            east_west_col_in=self.easting_col,
            north_south_col_in=self.northing_col,
            east_west_col_out="longitude",
            north_south_col_out="latitude",
        )

    def _set_up_intenseqc_framework(self) -> tuple[dict, list]:
        qc_kwargs = {
            "QC2": {"k": 10},
            "shared": {
                "time_res": self.time_res,
                "smallest_measurable_rainfall_amount": self.smallest_rainfall_amount,
                "wet_threshold": 1.0,
                "min_n_neighbours": self.min_n_neighbours,
                "n_neighbours_ignored": 0,
                "accumulation_multiplying_factor": 2.0,
            },
        }

        qc_methods_to_run = ["QC2", "QC10", "QC11", "QC12", "QC13", "QC14", "QC15", "QC17", "QC19", "QC20"]

        return qc_kwargs, qc_methods_to_run

    def _validate_input_crs(self, input_crs: str) -> str:
        assert input_crs.startswith("EPSG:"), (
            f"Invalid input_crs {input_crs}, needs to begin with 'EPSG:' like 'EPSG:4326'."
        )
        return input_crs

    def _validate_time_res(self, time_res: str) -> str:
        assert time_res in time_res_to_n_time_steps_in_day.keys(), (
            f"'{time_res}' not in accepted time resolutions for data. Accepted time res: {time_res_to_n_time_steps_in_day.keys()}"
        )
        return time_res

    @classmethod
    def run(
        cls, save_data: bool, return_data: bool, partition_by_columns: list = None, **kwargs
    ) -> None | tuple[pl.DataFrame, pl.DataFrame]:
        """
        Run the quality controller and return and/or save the prepared data.

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
        qc_data:
            Data run through algorithm
        qc_metadata:
            Metadata of data run through algorithm

        """
        quality_controller = cls(**kwargs)
        if quality_controller.verbose:
            print("Quality controlling data for gridder", flush=True)
        quality_controller.quality_control_data()
        if save_data:
            if quality_controller.verbose:
                print(f"Saving data to {quality_controller.output_dir}", flush=True)
            quality_controller.save_qcd_data(partition_by_columns)
            quality_controller.save_qcd_metadata()
            quality_controller.save_summary_of_qc()
            quality_controller.save_qc_rulebase_summary()
        else:
            if quality_controller.verbose:
                print("Data not saved", flush=True)
        if return_data:
            return (
                quality_controller.qcd_data,
                quality_controller.qcd_metadata,
                quality_controller.summary_of_qc,
                quality_controller.qc_rulebase_summary,
            )

    def get_nearest_neighbour(self, nearby_rainfall_data_loader, station_id):
        if len(nearby_rainfall_data_loader.nearby_rain_gauge_distances) > 0:
            return nearby_rainfall_data_loader.nearby_rain_gauge_distances.sort("distance")[0][
                self.station_id_col
            ].item()
        else:
            if self.verbose:
                print(f"Station ID: {station_id} has no neighbours\n", flush=True)
            return False

    def quality_control_data(self):
        # preallocate the output lists
        unique_station_ids = self.rainfall_metadata[self.station_id_col].unique()
        overall_summary_of_qc = [None] * len(unique_station_ids)
        qcd_data_list = [None] * len(unique_station_ids)
        rulebase_summary = [None] * len(unique_station_ids)

        # begin loop
        for ind, station_id in enumerate(unique_station_ids):
            nearby_rainfall_data_loader = NearbyRainfallDataLoader(
                metadata=self.rainfall_metadata,
                station_id=station_id,
                date_time_col=self.date_time_col,
                precipitation_col=self.precipitation_col,
                station_id_col=self.station_id_col,
                start_date_col=self.start_date_col,
                end_date_col=self.end_date_col,
                min_overlap_days=self.min_n_timesteps / time_res_to_n_time_steps_in_day[self.time_res],
                rainfall_data_source="df",
                rainfall_data_pl=self.rainfall_data,
                time_res=self.time_res,
                **self.nearby_rainfall_data_loader_kwargs,
            )

            # Check if that station actually has any neighbours
            if nearby_rainfall_data_loader.nearest_station_id is None:
                if self.verbose:
                    print(f"Station ID: {nearby_rainfall_data_loader.station_id} has no neighbours\n", flush=True)
                continue

            nearby_metadata = nearby_rainfall_data_loader.nearby_metadata
            nearby_rainfall_data = nearby_rainfall_data_loader.nearby_rainfall_for_rainfallqc

            # Update shared QC kwargs with latest values from nearby gauge loader
            self.update_shared_qc_kwargs(nearby_rainfall_data_loader)
            # Run QC framework
            try:
                assert len(nearby_rainfall_data) > self.min_n_timesteps, (
                    f"Data needs at least {self.min_n_timesteps} timesteps"
                )
                qc_result = rainfallqc.apply_qc_framework.run_qc_framework(
                    data=nearby_rainfall_data,
                    qc_framework=self.qc_framework,
                    qc_methods_to_run=self.qc_methods_to_run,
                    qc_kwargs=self.qc_kwargs,
                )
            except Exception as e:
                if self.verbose:
                    print(station_id, e, "\n", flush=True)
                continue

            # Summarise QC flags into statistics
            qc_summariser = QCSummariser(
                station_id=station_id,
                rainfall_data=nearby_rainfall_data,
                nearby_metadata=nearby_metadata,
                qc_result=qc_result,
                verbose=self.verbose,
            )

            # Apply rulebase
            rule_removed_rows, n_rows_removed = apply_intenseQC_rulebase(
                qc_summariser.all_flags, station_id, time_step=self.time_res
            )
            if self.verbose:
                print(
                    f"Station ID: {station_id}\tA total of {qc_summariser.all_flags['all_flags_by_row'][station_id].count() - rule_removed_rows[station_id].count()} rows were removed",
                    flush=True,
                )  # some rows may have stayed null
            ## get back into parquet format that fits with Oracle
            rule_removed_rows = rule_removed_rows.select(["time", station_id])  # saves memory
            rule_removed_rows = rule_removed_rows.with_columns(pl.lit(station_id).alias(self.station_id_col))
            rule_removed_rows = rule_removed_rows.rename(
                {station_id: self.precipitation_col, "time": self.date_time_col}
            )

            # Append summaries to lists
            overall_summary_of_qc[ind] = qc_summariser.summary_of_qc
            qcd_data_list[ind] = rule_removed_rows
            rulebase_summary[ind] = n_rows_removed
            if self.verbose:
                print("", flush=True)

        # Add summaries and qc data to self
        self.qc_rulebase_summary = pl.DataFrame(rulebase_summary)
        self.summary_of_qc = pl.DataFrame(overall_summary_of_qc)
        self.qcd_data = pl.concat([qcd_data for qcd_data in qcd_data_list if qcd_data is not None])

        # double check
        self.qcd_metadata = self.rainfall_metadata.filter(
            pl.col(self.station_id_col)
            .cast(pl.String)
            .is_in(self.summary_of_qc[self.station_id_col].drop_nulls().to_list())
        )

    def save_qcd_data(self, partition_by_columns: list = None) -> None:
        """
        Save data that has been quality controlled for gridding.

        Parameters
        ----------
        partition_by_columns:
            Columns that decide the partitioning of the output parquet file structure (default is station_id_col)
        """
        if partition_by_columns is None:
            partition_by_columns = [self.station_id_col]

        if self.qcd_data is None:
            raise RuntimeError("You must call quality_control_data() before save_qcd_data()")

        assert len(self.qcd_metadata.filter(pl.col("file_path").is_duplicated())) == 0, (
            "Problem with metadata as duplicate filepaths"
        )

        # Save partitioned parquet file
        (
            self.qcd_data.sort(self.date_time_col).write_parquet(
                self.output_dir / "qc_data",
                partition_by=partition_by_columns,
            )
        )
        if self.verbose:
            print(f"QC'd rainfall data available at: {self.output_dir / 'qc_data/'}", flush=True)

    def save_qcd_metadata(self) -> None:
        if self.qcd_metadata is None:
            raise RuntimeError("You must call quality_control_data() before save_final_metadata()")
        self.qcd_metadata.write_parquet(self.output_dir / "qcd_metadata.parquet")
        if self.verbose:
            print(f"QC'd rainfall metadata available at: {self.output_dir / 'qcd_metadata.parquet'}", flush=True)

    def save_summary_of_qc(self) -> None:
        if self.summary_of_qc is None:
            raise RuntimeError("You must call quality_control_data() before summary_of_qc()")
        self.summary_of_qc.write_parquet(self.output_dir / "summary_of_qc.parquet")
        if self.verbose:
            print(f"Summary of QC available at: {self.output_dir / 'summary_of_qc.parquet'}", flush=True)

    def save_qc_rulebase_summary(self) -> None:
        if self.qc_rulebase_summary is None:
            raise RuntimeError("You must call quality_control_data() before save_qc_rulebase_summary()")
        self.qc_rulebase_summary.write_parquet(self.output_dir / "qc_rulebase_summary.parquet")
        if self.verbose:
            print(f"Summary of QC rulebase available at: {self.output_dir / 'qc_rulebase_summary.parquet'}", flush=True)

    def update_shared_qc_kwargs(self, nearby_rainfall_data_loader: NearbyRainfallDataLoader) -> None:
        """
        Update all the shared keyword arguments.

        TODO: Check this updating in the loop properly.
        """
        self.qc_kwargs["shared"]["rain_col"] = nearby_rainfall_data_loader.station_id
        self.qc_kwargs["shared"]["target_gauge_col"] = nearby_rainfall_data_loader.station_id
        self.qc_kwargs["shared"]["nearest_neighbour"] = nearby_rainfall_data_loader.nearest_station_id
        self.qc_kwargs["shared"]["list_of_nearest_stations"] = nearby_rainfall_data_loader.nearby_rain_gauge_distances[
            self.station_id_col
        ].to_list()
        self.qc_kwargs["shared"]["gauge_lat"] = nearby_rainfall_data_loader.nearby_metadata.filter(
            pl.col(self.station_id_col) == nearby_rainfall_data_loader.station_id
        )["latitude"]
        self.qc_kwargs["shared"]["gauge_lon"] = nearby_rainfall_data_loader.nearby_metadata.filter(
            pl.col(self.station_id_col) == nearby_rainfall_data_loader.station_id
        )["longitude"]


class QCSummariser:
    """
    Summariser for QC flags.
    """

    def __init__(
        self,
        station_id: str,
        rainfall_data: pl.DataFrame,
        nearby_metadata: pl.DataFrame,
        qc_result: dict,
        verbose: bool,
    ):
        """
        Summarises QC from a given station.

        Parameters
        ----------
        station_id:
            Target rainfall station ID
        rainfall_data:
            Rainfall data from target ID
        nearby_metadata:
            Details about rainfall data from target and a given no. of neighbours
        qc_result:
            Summary of QC output from rainfallqc.apply_qc_framework.run_qc_framework.
        verbose:
            Whether to print progress as algorithm is run (default: False)

        """
        self.all_flags = {}
        self.station_id = station_id
        self.rainfall_data = rainfall_data
        self.nearby_metadata = nearby_metadata
        self.qc_result = qc_result
        self.verbose = verbose

        # Join QC checks into one dict
        self._join_qc_result_into_all_flags()
        # Get count of flagged and not-flagged rows
        flagged_rows, not_flagged_rows = self._count_flagged_and_non_flagged_rows()
        # Create summary
        self.all_flags, self.summary_of_qc = self._create_summary_of_qc(flagged_rows, not_flagged_rows=not_flagged_rows)

    def _join_qc_result_into_all_flags(self):
        self.all_flags["all_flags_by_row"] = self.rainfall_data["time", self.station_id]
        for qc in self.qc_result:
            if isinstance(self.qc_result[qc], pl.DataFrame):
                try:
                    self.all_flags["all_flags_by_row"] = self.all_flags["all_flags_by_row"].join(
                        self.qc_result[qc], on="time"
                    )
                except Exception as e:
                    if self.verbose:
                        print(e, self.station_id)
            else:
                self.all_flags[qc] = self.qc_result[qc]

    def _count_flagged_and_non_flagged_rows(self) -> tuple[pl.Series, pl.Series]:
        """
        Count number of rows of rainfall data with flags or not.
        """
        self.all_flags["all_flags_by_row"] = self.all_flags["all_flags_by_row"].with_columns(
            pl.when(
                pl.any_horizontal(
                    pl.all().exclude(["time", self.station_id]).fill_null(0.0).map_elements(lambda col: col > 0)
                )
            )
            .then(1)
            .otherwise(0)
            .alias("is_flagged")
        )

        # Count number of flagged rows
        flagged_rows = (
            self.all_flags["all_flags_by_row"]["is_flagged"].value_counts().filter(pl.col("is_flagged") == 1)["count"]
        )
        not_flagged_rows = (
            self.all_flags["all_flags_by_row"]["is_flagged"].value_counts().filter(pl.col("is_flagged") == 0)["count"]
        )
        return flagged_rows, not_flagged_rows

    def _create_summary_of_qc(self, flagged_rows: pl.Series, not_flagged_rows: pl.Series):
        """
        Create summary of the rows flagged in rainfall data.
        """
        total_rows = flagged_rows + not_flagged_rows
        perc_flagged = (flagged_rows / total_rows) * 100
        perc_flagged = perc_flagged.item() if perc_flagged.len() == 1 else 0
        if self.verbose:
            print(f"Station ID: {self.station_id}\t\tFlag rate: {perc_flagged: .2f}%", flush=True)

        # add to overall QC summary
        summary_of_qc = {}
        summary_of_qc["station_id"] = self.station_id
        summary_of_qc["num_nearby_gauges"] = len(self.nearby_metadata) - 1  # do not count the target
        summary_of_qc["perc_flagged"] = round(perc_flagged, 3)
        summary_of_qc["total_flagged_rows"] = flagged_rows[0] if flagged_rows.len() > 0 else 0
        summary_of_qc["total_rows"] = len(self.all_flags["all_flags_by_row"])

        for qc_key in NON_ROWWISE_QC_CHECKS:
            if qc_key not in self.all_flags:
                continue
            if isinstance(self.all_flags[qc_key], list):
                # sum number of years flagged
                summary_of_qc[NON_ROWWISE_QC_CONVERTER[qc_key]] = sum(item != 0 for item in self.all_flags[qc_key])
            else:
                summary_of_qc[NON_ROWWISE_QC_CONVERTER[qc_key]] = self.all_flags[qc_key]

        for col in self.all_flags["all_flags_by_row"].columns[2:]:
            summary_of_qc[col] = len(self.all_flags["all_flags_by_row"].filter(pl.col(col) > 0).drop_nans()[col])
        return self.all_flags, summary_of_qc
