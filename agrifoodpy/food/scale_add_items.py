import xarray as xr
import numpy as np
from agrifoodpy.pipeline import Node, standalone
from agrifoodpy.utils.scaling import logistic_scale

def scale_add_items_setup(datablock):
    return datablock

@standalone(input_keys=["dataset"], return_keys=["dataset"])
def scale_add_items_exec(dataset, in_array, out_array, items, scale,
                         add=True, timescale=None, start_year=None,
                         scale_func='logistic', datablock=None):
                         
    """Scales item quantities in one dataarray and adds the difference to
    another array in the same dataset.

    Parameters
    ----------
    dataset : dict
        Datablock path to the dataset to be scaled. Must point to an xarray
        Dataset.
    in_array : str
        Name of the array to scale.
    out_array : str
        Name of the array to add the difference to.
    items : list, tuple
        List of items to be scaled. If a tuple, the first element is the name of
        an item label coordinate, and the second element is the value or array
        of values used to select items.
    scale : float
        Fraction of the items to be replaced by the new items
    add : bool
        If True, the scaled items are added to the original items. If False,
        they are subtracted.
    aitional_datasets : list
        List of additional datasets to be scaled using the same resulting
        scaling ratio. Assumes these datasets have the same dimensions as the
        input dataset.
    """

    data = datablock[dataset].copy(deep=True)

    if isinstance(items, tuple):
        items = data.sel(Item = data[items[0]]==items[1]).Item.values

    if scale_func == "logistic":
        from agrifoodpy.utils.scaling import logistic_scale as scale_function
    elif scale_func == "linear":
        from agrifoodpy.utils.scaling import linear_scale as scale_function
    else:
        raise ValueError("Scale must be either 'logistic' or 'linear'")
    
    y0 = data.Year.values[0]
    y1 = start_year
    y3 = data.Year.values[-1]
    y2 = np.min([start_year + timescale, y3])
    
    scale = scale_function(y0, y1, y2, y3, c_init=1, c_end=scale)

    out = data.fbs.scale_add(
        element_in=in_array,
        element_out=out_array,
        scale=scale,
        items=items,
        add=add)

    # Obtain the ratio to scale add datasets
    ratio = out / data
    ratio = ratio.where(~np.isnan(ratio), 1)

    # if additional datasets are provided, scale them using the same ratio
    datablock[dataset] = data*ratio

    return datablock

scale_add_items = Node(scale_add_items_setup, scale_add_items_exec)