import fsspec
import xarray as xr
import zarr


def get_uk_mask_gear_coords():
    gear_daily = get_gear_daily()
    gear_daily = gear_daily.isel(time=0)
    return gear_daily["rainfall_amount"].notnull()


def get_uk_mask_haduk_coords(reverse_coords: bool=True) -> xr.Dataset:
    gear_daily = get_gear_daily()
    gear_daily = gear_daily.isel(time=0)
    gear_daily_haduk_coords = coerse_data_into_haduk_format(gear_daily, offset=-500)
    if reverse_coords:
        # Reverse y so ascending like CEH-GEAR
        gear_daily_haduk_coords = gear_daily_haduk_coords.sortby("y", ascending=True)
    return gear_daily_haduk_coords["rainfall_amount"].notnull()


def coerse_data_into_haduk_format(data: xr.Dataset, offset: int | float) -> xr.Dataset:
    """
    Quick fix for coersing data to have same grid as HADUK.
    """
    data = data.assign_coords(x=(data["x"] + offset))
    data = data.assign_coords(y=(data["y"] + offset))
    return data


def get_gear_daily():
    fdri_fs = fsspec.filesystem(
        "s3",
        asynchronous=True,
        anon=True,
        endpoint_url="https://fdri-o.s3-ext.jc.rl.ac.uk",
    )
    gear_daily_zstore = zarr.storage.FsspecStore(
        fdri_fs, path="geardaily/GB/geardaily_fulloutput_yearly_100km_chunks.zarr"
    )

    return xr.open_zarr(gear_daily_zstore, decode_times=True, decode_cf=True)  # 310 GB worth of data


def get_gear_hourly():
    fdri_fs = fsspec.filesystem(
        "s3",
        asynchronous=True,
        anon=True,
        endpoint_url="https://fdri-o.s3-ext.jc.rl.ac.uk",
    )
    gear_hourly_zstore = zarr.storage.FsspecStore(fdri_fs, path="gearhrly/gearhrly_15day_100km_chunks.zarr")

    return xr.open_zarr(gear_hourly_zstore, decode_times=True, decode_cf=True)
