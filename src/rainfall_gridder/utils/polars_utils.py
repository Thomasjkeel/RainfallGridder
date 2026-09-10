import polars as pl
import rainfallqc.utils.data_utils


def check_data_is_specific_time_res(data: pl.DataFrame, time_res: str | list) -> None:
    """
    Check data has a hourly or daily time step.

    Taken from RainfallQC.

    Does not work for monthly data, please use 'check_data_is_monthly'.

    Parameters
    ----------
    data :
        Data with time column.
    time_res :
        Time resolutions either a single string or list of strings

    Raises
    ------
    ValueError :
        If data is not hourly or daily.

    """
    return rainfallqc.utils.data_utils.check_data_is_specific_time_res(data, time_res)
