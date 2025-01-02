"""Pipeline utilities"""

import numpy as np

def item_parser(fbs, items):
    """Extracts a list of items from a dataset using a coordinate-key tuple,
    or converts a scalar item to a list

    Parameters
    ----------

    fbs : xarray.Dataset
        The FBS dataset

    items : tuple, scalar
        If a tuple, the first element is the name of the coordinate and the
        second element is a list of items to extract. If a scalar, the item
        is converted to a list.

    Returns
    -------
    list
        A list of items matching the coordinate-key description, or containing
        the scalar item.
    """

    if items is None:
        return None

    if isinstance(items, tuple):
        items = fbs.sel(Item = fbs[items[0]].isin(items[1])).Item.values
    elif np.isscalar(items):
        items = [items]

    return items

