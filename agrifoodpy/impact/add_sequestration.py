import xarray as xr
import numpy as np

def add_sequestration(datablock, type, max_seq, start_year, timescale,
                  food=None, scale_func="logistic"):
    
    """Adds total annual sequestration from different sources according to a 
    maximum annual sequestration and a scaling function.

    Parameters
    ----------
    datablock : dict
        The datablock dictionary.
    type : str
        The sequestration type name.
    max_seq : float
        The maximum annual sequestration in t CO2e/year.
    start_year : int
        The year to start the computation.
    timescale : int
        The timescale to use for the adoption scaling function.
    food : str
        The dataset key to use to extract year range from.
    scale_func : str
        The scaling function to use for the adoption scaling. Must be either
        'logistic' or 'linear'
    """

    if isinstance(timescale, str):
        timescale = datablock[timescale]

    food_orig = datablock[food]
    years = food_orig.Year.values

    # Make sue the type and max_seq are lists
    if np.isscalar(type):
        type = [type]

    if np.isscalar(max_seq):
        max_seq = [max_seq]

    for type, seq in zip(type, max_seq):
    # Compute forest area in ha, maximum anual sequestration, and growth curve

        if scale_func == "logistic":
            from agrifoodpy.utils.scaling import logistic_scale as scale_function
        elif scale_func == "linear":
            from agrifoodpy.utils.scaling import linear_scale as scale_function
        else:
            raise ValueError("Scale must be either 'logistic' or 'linear'")
        
        scale = scale_function(years[0], start_year, start_year+timescale,
                                        years[-1], c_init=0, c_end=1)

        sequestration = seq * scale

        # Create a dataset with the different sequestration sources
        seq_ds = xr.Dataset({type: sequestration})
    
        seq_da = seq_ds.to_array(dim="Item", name="sequestration")
    
        if "sequestration" not in datablock:
            datablock["sequestration"] = seq_da
        else:
            # append sequestration to existing sequestration da
            seq_da_in = datablock["sequestration"]
            seq_da = xr.concat([seq_da_in, seq_da], dim="Item")
            datablock["sequestration"] = seq_da

    return datablock
