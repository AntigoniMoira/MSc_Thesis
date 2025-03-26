import numpy as np
import xarray as xr
import os
import multiprocessing

def regridding_deg(ds_in):

    ds_out = ds_in.interp(latitude=ds_in.latitude[::2], longitude=ds_in.longitude[::2], method="cubic")
    
    return ds_out

def process_grib_file(year):
    # Define the input path where the GRIB files are located
    input_path = '/export/home/nfs/share/ERA5/20000101/'
    # output_path = '/export/home/nfs/Nevermore_DynDown/output_files'
    output_path = '/export/home/nfs/Nevermore_DynDown/output_files_deg_bicubic'
    # Construct the filename for the current year
    fname = str(year) + 'ALL_sfc.grib'
    out_fname = str(year) +'_t2m_sfc'
    
    # Open the GRIB file using xarray with cfgrib engine
    ds = xr.open_dataset(input_path + fname, engine='cfgrib')

    # print(ds)
    
    # Remove the first latitude and first longitude
    ds_filtered = ds.isel(latitude=slice(1, None), longitude=slice(1, None))

    # Create the new dataset
    ds_deg_0_25 = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), ds_filtered.t2m.values),
        },
        coords={
            "time": ds_filtered.time.values,
            "latitude": ds_filtered.latitude.values,
            "longitude": ds_filtered.longitude.values,
        },
    )
    
    # Save regridded data and corresponding index
    np.save(output_path+'/0_25x0_25/' + out_fname, ds_deg_0_25.t2m.values)
    np.save(output_path+'/0_25x0_25/index/' + out_fname + '_idx', ds.time.dt.strftime("%Y-%m-%d %H:%M").values)

    # Perform regridding to 0.5x0.5 grid
    ds_deg_0_5 = regridding_deg(ds_deg_0_25)
    
    # Save regridded data and corresponding index
    np.save(output_path+'/0_5x0_5/' + out_fname, ds_deg_0_5.t2m.values)
    np.save(output_path+'/0_5x0_5/index/' + out_fname + '_idx', ds.time.dt.strftime("%Y-%m-%d %H:%M").values)

    # Perform regridding to 1x1 grid
    ds_deg_1 = regridding_deg(ds_deg_0_5)
    
    # Save regridded data and corresponding index
    np.save(output_path+'/1x1/' + out_fname, ds_deg_1.t2m.values)
    np.save(output_path+'/1x1/index/' + out_fname + '_idx', ds.time.dt.strftime("%Y-%m-%d %H:%M").values)


if __name__ == '__main__':
  years = range(2000, 2021)
  for year in years:
      process_grib_file(year)