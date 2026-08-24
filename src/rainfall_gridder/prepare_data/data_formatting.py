import numpy as np
import polars as pl
import xarray as xr


def set_negative_precip_values_to_none(precip_data: pl.DataFrame, precip_col: str) -> pl.DataFrame:
    """
    Set values below 0 from the precip column to None.

    Parameters
    ----------
    precip_data:
        Data with precipitation column
    precip_col:
        Name of precipitation column

    Returns
    -------
    data_wo_non_neg:
        Data without non-negative precipitation values
    """
    return precip_data.with_columns(
        pl.when(pl.col(precip_col) < 0).then(None).otherwise(pl.col(precip_col)).alias(precip_col)
    )


def group_metadata_by_station_locations(metadata: pl.DataFrame, easting_col: str, northing_col: str) -> pl.DataFrame:
    """
    Group metadata by station locations and give a unique group ID.

    Parameters
    ----------
    metadata:
        Station metadata with easting and northing col
    easting_col:
        Name of easting coord column
    northing_col:
        Name of northing coord column

    Returns
    -------
    metadata_w_groupIDs:
        Metadata with station group ID column
    """
    return metadata.with_columns(pl.struct(easting_col, northing_col).rank(method="dense").alias("station_group_id"))


def add_blank_file_path_to_metadata(metadata: pl.DataFrame) -> pl.DataFrame:
    """
    Add empty file path column to metadata.

    Parameters
    ----------
    metadata:
        Station metadata

    Returns
    -------
    metadata_w_file_paths:
        Metadata with empty file_path column (string)
    """
    return metadata.with_columns(pl.lit(None).cast(pl.String).alias("file_path"))


def check_time_overlap_between_gridded_and_gauges(
    rainfall_data: pl.DataFrame,
    rainfall_date_time_col: str,
    gridded_rainfall: xr.Dataset,
    allow_imperfect_overlap: bool,
):
    """
    Check there is an overlap between the gridded and gauge data.

    Parameters
    ----------
    rainfall_data:
        (daily) Rainfall gauge data
    date_time_col:
        Name of date time col in rainfall data
    gridded_rainfall:
        Gridded rainfall data
    allow_imperfect_overlap:
        Whether to allow for an imperfect overlap between gridded and gauges (default False)

    Raises
    ------
    ValueError:
        If there is not an overlap, or there is an overlap but it does not cover the gauge record and allow_imperfect_overlap is False.]
        Or if the overlap is less than 50% of total rows

    """
    rainfall_data_time_min = rainfall_data[rainfall_date_time_col].min()
    rainfall_data_time_max = rainfall_data[rainfall_date_time_col].max()
    gridded_rainfall_time_min = pl.from_numpy(np.array([gridded_rainfall["time"].min().values])).item()
    gridded_rainfall_time_max = pl.from_numpy(np.array([gridded_rainfall["time"].max().values])).item()

    gridded_rainfall_overlap = gridded_rainfall.sel(time=slice(rainfall_data_time_min, rainfall_data_time_max))

    if gridded_rainfall_overlap["time"].size < 1:
        raise ValueError(
            f"No overlap between rain gauge data (runs from {rainfall_data_time_min} to {rainfall_data_time_max}) and gridded rainfall (run from {gridded_rainfall_time_min} to {gridded_rainfall_time_max})."
        )

    if allow_imperfect_overlap:
        overlap_start = max(rainfall_data_time_min, gridded_rainfall_time_min)
        overlap_end = min(rainfall_data_time_max, gridded_rainfall_time_max)

        if overlap_start > overlap_end:
            raise ValueError(
                "No overlap between gridded rainfall data and rain gauge data. "
                f"Rainfall data: {rainfall_data_time_min} to {rainfall_data_time_max}; "
                f"gridded data: {gridded_rainfall_time_min} to {gridded_rainfall_time_max}."
            )

        overlap_days = (overlap_end - overlap_start).days + 1  # plus 1d because inclusive
        rainfall_total_days = (rainfall_data_time_max - rainfall_data_time_min).days + 1  # plus 1d because inclusive
        if overlap_days != rainfall_total_days:
            # Check for at least 50% overlap in days
            if overlap_days <= (rainfall_total_days / 2):
                raise ValueError(
                    "Not enough overlap between gridded rainfall data and rain gauge data. "
                    f"Rainfall data: {rainfall_data_time_min} to {rainfall_data_time_max}; "
                    f"gridded data: {gridded_rainfall_time_min} to {gridded_rainfall_time_max}."
                )
            else:
                print(
                    f"Warning: imperfect overlap (overlap {overlap_days}/{rainfall_total_days} days) between gridded rainfall data and rain gauge data. "
                    f"Rainfall data: {rainfall_data_time_min} to {rainfall_data_time_max}; "
                    f"gridded data: {gridded_rainfall_time_min} to {gridded_rainfall_time_max}."
                )

    else:
        if gridded_rainfall_time_min > rainfall_data_time_min or gridded_rainfall_time_max < rainfall_data_time_max:
            raise ValueError(
                "No or imperfect overlap between gridded rainfall data and rain gauge data. "
                f"Rain gauge data: {rainfall_data_time_min} to {rainfall_data_time_max}; "
                f"gridded data: {gridded_rainfall_time_min} to {gridded_rainfall_time_max}."
                "You can allow imperfect overlap by setting `allow_imperfect_overlap=True`"
            )
