import xarray as xr
import numpy as np

from agrifoodpy.pipeline import standalone

@standalone(["impact"], ["impact"])
def scale_impact(impact, scale_factor, items=None, timescale=None,
                 start_year=None, scale_func='logistic', datablock=None):
    """Scales impact quantities by a multiplicative factor for selected items.

    Parameters
    ----------
    impact : str
        Datablock key for the impact dataset, or the impact dataset itself.
    scale_factor : float
        Multiplicative factor to scale the impact quantities by.
    items : list
        List of items to scale.
    timescale : int, str
        Datablock key for the timescale dataset, or the timescale in years
        itself
    start_year : int
        Year to start the scaling from.
    scale_func : str
        Function to use for scaling. Can be 'logistic' or 'linear'.
    """

    # load impacts
    data = datablock[impact].copy(deep=True)

    # if no items are specified, scale all items
    if items is None:
        items = data.Item.values

    # if items is a tuple, extract item list using (label) coordinates
    elif isinstance(items, tuple):
        items = data.sel(Item = data[items[0]]==items[1]).Item.values

    # scale the impacts
    if scale_func == 'logistic':
        from agrifoodpy.utils.scaling import logistic_scale as scale_func
    elif scale_func == 'linear':
        from agrifoodpy.utils.scaling import linear_scale as scale_func
    else:
        raise ValueError("scale_func must be one of 'logistic' or 'linear'")

    if isinstance(timescale, str):
        timescale = datablock[timescale]

    y0 = data.Year.values[0]
    y1 = start_year
    y3 = data.Year.values[-1]
    y2 = np.min([start_year + timescale, y3])

    scale_arr = scale_func(y0, y1, y2, y3, c_init=1, c_end=scale_factor)

    data.loc[{"Item": items}] *= scale_arr

    datablock[impact] = data

    return datablock
