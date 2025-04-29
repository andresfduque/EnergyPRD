import xarray as xr
import datetime as dt
from itertools import chain
import numpy as np

def retrieve_gfs_dods(lat, lon, start_date, end_date, var_dict, runs = 0):
    """_summary_

    Args:
        lat (tuple): min and max latitude
        lon (tuple): min and max longitude
        time (_type_): _description_
        var_dict (_type_): _description_
        runs (int, optional): _description_. Defaults to 0.
    """    

lat = (-5, 5)
lon = (-76, -75)
time = (0, 1)
var = ["tmp2m"]
runs = [0]


"https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs20230702/gfs_0p25_1hr_00z"



for x in range(0,1):
    url = 'https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs20220815/gfs_0p25_1hr_06z'
    url = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.1981.days_p05.nc'
    print(url)
    with xr.open_dataset(url) as ds:
        (
            ds[var]
            .isel(time=slice(*time))
            .sel(lat=slice(*lat), lon=lon)
            .to_netcdf("/home/andresd/Downloads/Test/test.nc")
        )


with xr.open_dataset(url) as ds:
    da = ds[var].isel(time = 0).sel(lat = slice(*lat), lon = slice(*lon))
    
    # this downloads data
    lons = da.lon.data
    lats = da.lat.data
    data = da.data
    da.to_netcdf('/home/andresd/Downloads/Test/test.nc')