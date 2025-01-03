import numpy as np
import xarray as xr

from agrifoodpy.pipeline import standalone

@standalone(input_keys=["fbs"], return_keys=["fbs"])
def fbs_convert(fbs, convertion_arr, keys=None, datablock=None):
    """Converts quantities in the food balance sheet using a conversion
    dataarray, dataset, or scaling factor.
    
    Parameters
    ----------
    datablock : Dict
        Dictionary containing data.
    dataset : str, xarray.Dataset
        Datablock paths to the food balance sheet datasets or the datasets
        themselves.
    convertion_arr : str, xarray.DataArray, tuple
        Datablock path to the conversion array, dataset-key tuple, or the array
        itself.
    keys : str, list
        Datablock key of the resulting dataset to be stored in the datablock.

    Returns
    -------
    dict or xarray.Dataset
        - Updated datablock if  a datablock is provided.
        - xarray.Dataset with converted quantities if no datablock is provided.
    """

    data = datablock[fbs]    

    # Prepare convertion array
    if isinstance(convertion_arr, str):
        convertion_arr = datablock[convertion_arr]
    
    elif isinstance(convertion_arr, tuple):
        convertion_arr = datablock[convertion_arr[0]][convertion_arr[1]]

    if isinstance(convertion_arr, xr.DataArray):
        convertion_arr = convertion_arr.where(np.isfinite(convertion_arr), other=0)

    # If no key is provided, overwrite original dataset
    if keys is None:
        keys = fbs

    datablock[keys] = data*convertion_arr

    return datablock
