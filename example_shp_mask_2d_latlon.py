import numpy as np
import xarray as xr
import geopandas as gpd
from mask_data_with_shapefile import shp_mask_2d_latlon

# Read in data
sample_dir = '/home/data_location'
variable_name = 'sample_name'
da = xr.open_dataset(f'{sample_dir}/datafile.nc')[variable_name]

# Read in shapefile
shp_dir = '/home/shp_location'
shp = gpd.read_file(f'{shp_dir}/shapefile.shp')

# Mask da with shp
da_mask = shp_mask_2d_latlon(data=da,lat=da.XLAT,lon=da.XLONG,shp_type='gdf',shp_file=shp,
                             lat_name='XLAT',lon_name='XLONG',reduce=True)
