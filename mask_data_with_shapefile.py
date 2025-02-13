######################################################################################
#
# These functions mask a data array based on an input shapefile 
#
# #####################################################################################

import numpy as np                      # works with v1.23.5
import xarray as xr                     # works with v2023.1.0
import geopandas as gpd                 # works with v0.14.2
import rioxarray                        # works with v0.16.0                 
from shapely.vectorized import contains # works with shapely v2.0.1
from shapely.geometry import mapping
from typing import Union

######################################################################################
# Create mask from shapefile when netcdf data has 1D lat/lon dimensions
# #####################################################################################

def shp_mask_1d_latlon(### Required inputs
                       data: xr.DataArray, shp_type: str,
                       
                       ### Optional inputs
                       shp_gdf: gpd.geodataframe.GeoDataFrame = None, shp_path: str = '',
                       crs: str = 'epsg:4326', lon_0_360: bool = False,
                       lat_name: str = 'lat', lon_name: str = 'lon'

                       ) -> xr.DataArray:

   '''Takes an unspecified xarray DataArray and creates a mask based on a specified 
      shapefile.

      Required inputs
      ------------------------------------------------------------------------------
      data
       class: 'xarray.DataArray', DataArray that will be masked by this function, 
                                  requires dimensions for at least lat and lon.
      shp_type
       class: 'string', This function allows for either a pre-loaded geodataframe 
                        ('gdf') to be used or a path to be provided that the 
                        function itself uses to load a shapefile ('path')
                        Options = 'gdf', 'path'

      Optional inputs 
      ------------------------------------------------------------------------------
      shp_gdf: class 'geopandas.geodataframe.GeoDataFrame', If shp_type = 'gdf',
               this variable is set as the shapefile. Choose this option if the 
               shapefile needs to be modified and modify before adding to function.
      shp_path: class 'string', If shp_type = 'path', this is the path to the 
                shapefile. Choose this option if the shapefile does not need to be
                modified.
      crs: class 'string', The coordinate reference system. Default is 'epsg:4326',
                           which is the identifier for WGS84.
      lon_0_360: class 'bool', If the data's lon coordinate has values from 0 to 360,
                               as opposed to -180 to 180, set this parameter to True.
                               The lon will then be transformed to -180 to 180 for the
                               masking to occur.
      lat_name, lon_name: class 'string', The name of the lat/lon dimensions of the
                                          DataArray.
      ------------------------------------------------------------------------------

      Returns
      -------------------------
      output: xarray.DataArray (masked DataArray)

   '''

   # Store original data's lat/lon dimensions
   latarr, lonarr = data[lat_name], data[lon_name] 

   # If lon_0_360 == True, transform lon values to be from -180 to 180
   if lon_0_360 == True:

     # Transform lonarr values
     lonarr.coords[lon_name] = ((lonarr.coords[lon_name] + 180) % 360) - 180
     lonarr = lonarr.sortby(lonarr[lon_name])
     lonarr = xr.where(lonarr >= 180, lonarr - 360, lonarr)

     # Transform data's lon values
     data.coords[lon_name] = (data.coords[lon_name] + 180) % 360 - 180
     data = data.sortby(data[lon_name])

   # Define shapefile
   if shp_type == 'gdf': 

    # Read in pre-defined shapefile
    shp = shp_gdf 

   elif shp_type == 'path': 

    # Read in shapefile from file path
    shp = gpd.read_file(shp_path,crs=crs)

   # Set spatial dimensions for data
   data.rio.set_spatial_dims(x_dim=lon_name,y_dim=lat_name,inplace=True)
   data.rio.write_crs(crs,inplace=True)

   # Mask the DataArray data with the shapefile
   mask = data.rio.clip(shp.geometry.apply(mapping),shp.crs,drop=True)

   return mask

######################################################################################
# Create mask from shapefile when input data array has 2D lat/lon dimensions
# #####################################################################################

def shp_mask_2d_latlon(### Required inputs
                       data: Union[xr.DataArray, np.ndarray],
                       lat: Union[xr.DataArray, np.ndarray], lon: Union[xr.DataArray, np.ndarray],
                       shp_type: str,

                       ### Optional inputs
                       shp_gdf: gpd.geodataframe.GeoDataFrame = None, shp_path: str = '',
                       reduce: bool = True, output: str = 'xarray',
                       lat_name: str = 'lat', lon_name: str = 'lon'
                       ):

    '''Takes a data array whose lat/lon dimensions are 2-dimensional and creates a
       mask based on a specified shapefile.

       Required inputs
       ------------------------------------------------------------------------------
       data
        class: 'numpy.ndarray' or 'xarray.DataArray', Data array that will be masked
                by this function. Can be either a numpy or xarray data array.
       shp_type
        class: 'string', This function allows for either a pre-loaded shapefile to be
                         used ('file') or a path to be provided that the function
                         itself uses to load a shapefile ('path')
                         Options = 'file', 'path'

       Optional inputs
       ------------------------------------------------------------------------------
       shp_gdf: class 'geopandas.geodataframe.GeoDataFrame', If shp_type = 'gdf',
               this variable is set as the shapefile. Choose this option if the 
               shapefile needs to be modified and modify before adding to function.
       shp_path: class 'string', If shp_type = 'path', this is the path to the
                 shapefile. Choose this option if the shapefile does not need to be
                 modified.
       reduce: class 'bool', Specifies if data should be reduced when output. If
                             True, then all rows/columns without a single value in
                             the shapefile area will be removed. Default = True
       output: class 'string', Data type of the output variables. Can be either
                               'numpy' to output data as a numpy array or 'xarray'
                               to output data as an xarray data array. If outputting
                               as an xarray data array, 'data' must also be an
                               xarray data array. Default = 'xarray'
       lat_name, lon_name: class 'string', The name of the lat/lon dimensions of the
                                           DataArray.
       ------------------------------------------------------------------------------

       Returns
       -------------------------
       output: data (masked), lat, lon

    '''

    # Convert data, lat, and lon to numpy arrays
    da_data = data                       # xarray placeholder variable
    data, lat, lon = np.array(data), np.array(lat), np.array(lon)
    
    # Define shapefile
    if shp_type == 'gdf':
        # Read in pre-defined shapefile
        shp = shp_gdf
    elif shp_type == 'path':
        # Read in shapefile from file path
        shp = gpd.read_file(shp_path)
    
    # If multiple geometries exist in shapefile, join them together
    if len(shp.geometry) == 1:
        geometry = shp.geometry.iloc[0] # if only one geometry, use it 
    elif len(shp.geometry) > 1:
        geometry = shp.geometry.unary_union # if multiple geometries, join them
    
    # Create mask of data based on shapefile (in the mask = 1., out of the mask = NaN)
    mask = np.where(contains(geometry,lon.flatten(),lat.flatten()).reshape(lon.shape),1,np.nan)
    
    # Apply mask to data
    data_mask = data * mask
    
    # Output the masked data
    if output == 'xarray': # convert numpy arrays to xarray data arrays
    
        # If outputting to xarray, input data must also be xarray, Print error message if not the case
        if not isinstance(da_data,xr.DataArray):
            print('Input data must be xr.DataArray for this output option')
    
        # If not reducing data, simple process to convert numpy arrays back to xarray data arrays
        if reduce == False:
            lat = xr.DataArray(lat,dims=da_data[lat_name].dims,coords=da_data[lat_name].coords)
            lon = xr.DataArray(lon,dims=da_data[lon_name].dims,coords=da_data[lon_name].coords)
            data_mask = xr.DataArray(data_mask,dims=da_data.dims,coords={lat_name: lat,lon_name: lon})
        
        # Reduce data volume by removing entire rows/columns not within the shapefile area
        # But more care needs to be taken if converting the reduced data arrays back to xarray
        else: # reduce == True
            
            # Define elements to keep for each row and column
            rows_to_keep = np.arange(data_mask.shape[0])[~np.isnan(data_mask).all(axis=1)]
            cols_to_keep = np.arange(data_mask.shape[1])[~np.isnan(data_mask).all(axis=0)]
    
            # Apply to ndarrays to only save rows/cols with relevant data for new area and convert to xarray
            # Convert reduced data arrays
            lat = xr.DataArray(da_data[lat_name].values[np.ix_(rows_to_keep,cols_to_keep)],name=lat_name,dims=da_data[lat_name].dims,
                               coords={lat_name: (da_data[lat_name].dims, da_data[lat_name].values[np.ix_(rows_to_keep,cols_to_keep)]),
                                       lon_name: (da_data[lon_name].dims, da_data[lon_name].values[np.ix_(rows_to_keep,cols_to_keep)])})
            lon = xr.DataArray(da_data[lon_name].values[np.ix_(rows_to_keep,cols_to_keep)],name=lon_name,dims=da_data[lon_name].dims,
                               coords={lat_name: (da_data[lat_name].dims, da_data[lon_name].values[np.ix_(rows_to_keep,cols_to_keep)]),
                                       lon_name: (da_data[lon_name].dims, da_data[lon_name].values[np.ix_(rows_to_keep,cols_to_keep)])})
            data_mask = xr.DataArray(data_mask[np.ix_(rows_to_keep,cols_to_keep)],dims=da_data.dims,
                                     coords={lat_name:lat,lon_name:lon})
    
    elif output == 'numpy':
    
        # Reduce data volume by removing entire rows/columns not within the shapefile area
        if reduce == True:
    
            # Define elements to keep for each row and column
            rows_to_keep = np.arange(data_mask.shape[0])[~np.isnan(data_mask).all(axis=1)]
            cols_to_keep = np.arange(data_mask.shape[1])[~np.isnan(data_mask).all(axis=0)]
        
            # Apply to data_mask to only save rows/cols with relevant data for new area
            data_mask = data_mask[np.ix_(rows_to_keep,cols_to_keep)]
            lat = lat[np.ix_(rows_to_keep,cols_to_keep)]
            lon = lon[np.ix_(rows_to_keep,cols_to_keep)]
    
    return data_mask, lat, lon
