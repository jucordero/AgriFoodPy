import xarray as xr
import numpy as np

from agrifoodpy.pipeline import standalone

@standalone(["land"], ["output_key"])
def land_sequestration(land, land_type, land_scale, years,
                       output_key="land_sequestration", start_year=None,
                       timescale=None, scale_func="logistic",
                       datablock=None):
    
    """Computes total quantities from different land types
    
    Parameters
    ----------
    land : str, xarray.DataArray
        Datablock Key for the land dataset, or the dataset itself.
    land_type : str
        The land types to compute quantities for.
    scale : float
        Annual scale factor for the land quantities.
    years : array, xarray.DataArray, xarray.DataSet, str
        Year array to use for the output dataset. If a datablock key is given,
        the year range of the dataset at the given key is used. If an xarray
        DataArray or xarray.DataSet is given, their year range is used.
    start_year : int
        The year to start the computation.
    timescale : int
        The timescale to use for the adoption scaling function.
    scale_func : str
        The scaling function to use for the adoption scaling. Must be either
        'logistic' or 'linear'
    datablock : dict
        The datablock dictionary.
    """

    if isinstance(timescale, str):
        timescale = datablock[timescale]

    # Make sue the type and max_seq are lists
    if np.isscalar(land_type):
        land_type = [land_type]

    if np.isscalar(land_scale):
            land_scale = [land_scale]

    if isinstance(years, str):
        years = datablock[years].Year.values
    elif isinstance(years, (xr.DataArray, xr.Dataset)):
        years = years.Year.values

    if scale_func == "logistic":
        from agrifoodpy.utils.scaling import logistic_scale as scale_function
    elif scale_func == "linear":
        from agrifoodpy.utils.scaling import linear_scale as scale_function
    else:
        raise ValueError("Scale must be either 'logistic' or 'linear'")
    
    scale = scale_function(years[0], start_year, start_year+timescale,
                                    years[-1], c_init=0, c_end=1)
    
    # Load the land use data from the datablock
    pctg = datablock[land].copy(deep=True)

    for lt, seq in zip(land_type, land_scale):
    # Compute forest area in ha, maximum anual sequestration, and growth curve
        land_area = pctg.loc[{"aggregate_class":lt}].sum().to_numpy()

        max_seq = land_area * seq
        seq_arr = max_seq * scale

        # Create a dataset with the different sequestration sources
        seq_ds = xr.Dataset({lt: seq_arr})
    
        seq_da = seq_ds.to_array(dim="Item", name=output_key)
    
        if output_key not in datablock:
            datablock[output_key] = seq_da
        else:
            # append sequestration to existing sequestration da
            seq_da_in = datablock[output_key]
            seq_da = xr.concat([seq_da_in, seq_da], dim="Item")
            datablock[output_key] = seq_da

    return datablock
