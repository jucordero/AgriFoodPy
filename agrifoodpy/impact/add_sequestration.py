import xarray as xr
import numpy as np

from agrifoodpy.pipeline import standalone

@standalone([], ["output"])
def add_sequestration(type, max_seq, years, output="output", 
                      start_year=None, timescale=None,
                      scale_func="logistic", datablock=None):
    """Adds total annual sequestration from different sources according to a 
    maximum annual sequestration and a scaling function.

    Parameters
    ----------
    seq : str
        Datablock key for the sequestration dataset.
    type : str
        Sequestration type to be used as DataArray name.
    max_seq : float
        The maximum annual sequestration in t CO2e/year.
    years : array, xarray.DataArray, xarray.DataSet, str
        Year array to use for the output dataset. If a datablock key is given,
        the year range of the dataset at the given key is used. If an xarray
        DataArray or xarray.DataSet is given, their year range is used.
    sequestration : str
        Datablock key to store the sequestration dataset.
    start_year : int
        Year to start the computation.
    timescale : int
        Timescale to use for the adoption scaling function.
    scale_func : str
        Scaling function to use for the adoption scaling. Must be either
        'logistic' or 'linear'
    datablock : dict
        The datablock dictionary.
    """

    if isinstance(timescale, str):
        timescale = datablock[timescale]

    # Make sue the type and max_seq are lists
    if np.isscalar(type):
        type = [type]

    if np.isscalar(max_seq):
        max_seq = [max_seq]

    # Load scaling function
    if scale_func == "logistic":
        from agrifoodpy.utils.scaling import logistic_scale as scale_function
    elif scale_func == "linear":
        from agrifoodpy.utils.scaling import linear_scale as scale_function
    else:
        raise ValueError("Scale must be either 'logistic' or 'linear'")
    
    if isinstance(years, str):
        years = datablock[years].Year.values
    elif isinstance(years, [xr.DataArray, xr.Dataset]):
        years = years.Year.values
    
    scale = scale_function(years[0], start_year, start_year+timescale,
                                        years[-1], c_init=0, c_end=1)
    
    # Write sequestration to datablock
    for type, seq in zip(type, max_seq):

        seq_arr = seq * scale

        # Create a dataset with the different sequestration sources
        seq_ds = xr.Dataset({type: seq_arr})
    
        seq_da = seq_ds.to_array(dim="Item", name=output)
    
        if output not in datablock:
            datablock[output] = seq_da
        else:
            # append sequestration to existing sequestration da
            seq_da_in = datablock[output]
            seq_da = xr.concat([seq_da_in, seq_da], dim="Item")
            datablock[output] = seq_da

    return datablock
