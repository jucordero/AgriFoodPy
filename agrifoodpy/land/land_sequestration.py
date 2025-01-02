import xarray as xr
import numpy as np

from agrifoodpy.pipeline import Node

def land_seq_setup(datablock):
    return datablock

def land_seq_exec(datablock, land, land_type, seq_ha_yr, start_year, timescale,
                  food=None, scale_func="logistic"):
    
    """Computes total annual sequestration from different land types
    
    Parameters
    ----------
    datablock : dict
        The datablock dictionary.
    land : str
        Key for the land dataset to use for the computation.
    land_type : str
        The land types to be compute sequestrations for.
    seq_ha_yr : float
        Annual sequestration in t CO2e/ha/year.
    start_year : int
        The year to start the computation.
    timescale : int
        The timescale to use for the adoption scaling function.
    food : str
        The food dataset to use to extract year range from.
    scale_func : str
        The scaling function to use for the adoption scaling. Must be either
        'logistic' or 'linear'
    """

    if isinstance(timescale, str):
        timescale = datablock[timescale]

    # Make sue the type and max_seq are lists
    if np.isscalar(land_type):
        land_type = [land_type]

    if np.isscalar(seq_ha_yr):
            seq_ha_yr = [seq_ha_yr]

    food_orig = datablock[food]
    years = food_orig.Year.values

    # Load the land use data from the datablock
    pctg = datablock[land].copy(deep=True)

    for lt, seq in zip(land_type, seq_ha_yr):
    # Compute forest area in ha, maximum anual sequestration, and growth curve
        land_area = pctg.loc[{"aggregate_class":lt}].sum().to_numpy()

        max_seq = land_area * seq

        if scale_func == "logistic":
            from agrifoodpy.utils.scaling import logistic_scale as scale_function
        elif scale_func == "linear":
            from agrifoodpy.utils.scaling import linear_scale as scale_function
        else:
            raise ValueError("Scale must be either 'logistic' or 'linear'")
        
        scale = scale_function(years[0], start_year, start_year+timescale,
                                        years[-1], c_init=0, c_end=1)

        sequestration = max_seq * scale

        # Create a dataset with the different sequestration sources
        seq_ds = xr.Dataset({lt: sequestration})
    
        seq_da = seq_ds.to_array(dim="Item", name="sequestration")
    
        if "sequestration" not in datablock:
            datablock["sequestration"] = seq_da
        else:
            # append sequestration to existing sequestration da
            seq_da_in = datablock["sequestration"]
            seq_da = xr.concat([seq_da_in, seq_da], dim="Item")
            datablock["sequestration"] = seq_da

    return datablock

land_sequestration = Node(land_seq_setup, land_seq_exec)