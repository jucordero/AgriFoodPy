""" Module for land intervention models
"""

from agrifoodpy.pipeline import standalone
import numpy as np
import xarray as xr

@standalone(["land", "mask"], ["land"])
def land_repurposing(land, land_type, fraction, new_types,
                          ratio=None, mask=None, mask_values=None,
                          datablock=None):
    """Replaces a fraction of an input list of land types by a new or existing
    type, assuming each pixel contains percentage data".

    Parameters
    ----------
    land : str
        Datablock path to the dataset containing the land use data.
    land_type : str, list
        The types of land to be replaced.
    fraction : float
        Fraction of the land type to be replaced.
    new_types : list
        List of new land types to be added to the land use dataset.
    ratio : float
        Relative proportion of the new land types to be added to the new land
        use type, with respect to the total repurposed land.
    """

    # Create datablock if it not provided
    standalone = False
    if datablock is None:
        datablock = {
            "land" : land,
        }
        land = "land"
        if mask is not None:
            datablock["mask"] = mask
            mask = "mask"
        standalone = True

    # Load land use data from datablock
    pctg = datablock[land].copy(deep=True)

    # Ensure land_type, new_types and ratio are arrays
    if np.isscalar(new_types):
        new_types = [new_types]

    ratio = np.array(ratio) if ratio is not None else np.ones(len(new_types))

    if np.isscalar(land_type):
        land_type = [land_type]

    # Check if new_types and ratio have the same number of elements
    if len(new_types) != len(ratio):
        raise ValueError("new_types and ratio must have the same number of elements")

    # Normalize ratio to sum to 1
    ratio = ratio / np.sum(ratio)

    # if no mask array is provided, then use the whole map
    for lt in land_type:

        if mask is not None:
            mask_arr = datablock[mask]
            mask_arr_data = np.isin(mask_arr, mask_values)
        else:
            mask_arr_data = np.ones_like(pctg, dtype=bool)

        to_repurpose = pctg.where(mask_arr_data, other=0).sel({"aggregate_class":lt})

        # Compute spared fraction to be re forested and remove from the spared class
        delta_repurposed = to_repurpose * fraction
        pctg.loc[{"aggregate_class":lt}] -= delta_repurposed

        for type, r in zip(new_types, ratio):
            if type not in pctg.aggregate_class.values:
                new_class = xr.zeros_like(pctg.isel(aggregate_class=0)).where(np.isfinite(pctg.isel(aggregate_class=0)))
                new_class["aggregate_class"] = type
                pctg = xr.concat([pctg, new_class], dim="aggregate_class")
            pctg.loc[{"aggregate_class":type}] += delta_repurposed * r

    # Rewrite land use data to datablock
    datablock[land] = pctg

    if standalone:
        return datablock[land]

    return datablock


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