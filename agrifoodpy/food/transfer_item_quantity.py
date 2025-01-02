import xarray as xr
import numpy as np
import copy

from agrifoodpy.pipeline import Node
from agrifoodpy.pipeline.utils import item_parser
from agrifoodpy.utils.scaling import logistic_scale

from agrifoodpy.pipeline import standalone

def transfer_item_quantity_setup(datablock):
    return datablock

@standalone(input_keys=["dataset"], return_keys=["dataset"])
def transfer_item_quantity_exec(dataset, element, item, item_out, scale, source,
                                fallback=None, timescale=None, start_year=None,
                                scale_feed_seed=False, datablock=None):
    """Reduces a fraction of the total quantity of an item and adds that
    difference to another existing item in a food balance sheet, scaling their
    production or imports accordingly.

    Parameters
    ----------
    datablock : dict
        Datablock containing pipeline data
    dataset : str, xarray.Dataset
        Datablock key to the Food Balance Sheet to be scaled, or the dataset
        itself
    element : str
        Array name for the element to be scaled
    item : list, tuple
        List of item to be scaled
    item_out : list, tuple
        Item or items for scaled difference quantity to be added to
    scale : float
        Fraction of item to be transferred
    source : str
        Source of item to be scaled
    fallback : str
        Element to be used as fallback if the source element results in a
        negative quantity. If None, negative values are left unchanged.
    timescale : int
        Time period over which the scaling is applied
    start_year : int
        Year from which the scaling is applied
    """

    # Read adoption timescale 
    if isinstance(timescale, str):
        timescale = datablock[timescale]

    food_orig = datablock[dataset]
    if isinstance(item, tuple):
        items_to_replace = food_orig.sel(Item=food_orig[item[0]].isin(item[0])).Item.values
    else:
        items_to_replace = item

    y0 = food_orig.Year.values[0]
    y1 = start_year
    y2 = np.min([start_year + timescale, food_orig.Year.values[-1]])
    y3 = food_orig.Year.values[-1]

    scale_labmeat = logistic_scale(y0, y1, y2, y3, c_init=1, c_end=scale)
    
    # Scale and remove from suplying element
    out = food_orig.fbs.scale_add(element_in=element,
                                element_out=source,
                                scale=scale_labmeat,
                                items=items_to_replace,
                                add=True)
    
    if scale_feed_seed:
        out = _feed_seed_processing_scale(out, food_orig,
                                            items_feed=("Item_origin", "Animal Products"),
                                            items_seed=("Item_origin", "Vegetal Products"),
                                            items_processing=out.Item.values)

    # If production is negative, set to zero and add the negative delta to
    # imports
    if fallback is not None:
        out = check_negative_source(out, source, fallback)
    
    # Add delta to cultured meat
    delta = (food_orig-out).sel(Item=items_to_replace).expand_dims("Item").sum(dim="Item")
    out.loc[{"Item":item_out}] += delta
    datablock[dataset] = out

    return datablock

def check_negative_source(fbs, source, fallback):
    """Checks for negative values in the source element and adds the difference
    to the fallback element"""

    delta_neg = fbs[source].where(fbs[source] < 0, other=0)
    fbs[source] -= delta_neg
    fbs[fallback] += delta_neg

    return fbs

def _feed_seed_processing_scale(fbs, reference, items_feed=None,
                                items_seed=None, items_processing=None,
                                feed="feed", seed="seed",
                                processing="processing",
                                production="production"):
    """Scales the feed, seed and processing quantities according to the change
    in production of specific items"""

    items_feed = item_parser(fbs, items_feed)
    items_seed = item_parser(fbs, items_seed)
    items_processing = item_parser(fbs, items_processing)

    if items_feed is not None:
        feed_scale = fbs[production].sel(Item=items_feed).sum(dim="Item") \
                    / reference[production].sel(Item=items_feed).sum(dim="Item")

        out = fbs.fbs.scale_add(element_in=feed, element_out=production,
                                scale=feed_scale)

    if items_seed is not None:
        seed_scale = fbs[production].sel(Item=items_seed).sum(dim="Item") \
                    / reference[production].sel(Item=items_seed).sum(dim="Item")
                
        out = out.fbs.scale_add(element_in=seed, element_out=production,
                                scale=seed_scale)    

    if items_processing is not None:
        processing_scale = fbs[production].sel(Item=items_processing).sum(dim="Item") \
                    / reference[production].sel(Item=items_processing).sum(dim="Item")

        out = out.fbs.scale_add(element_in=processing, element_out=production,
                                scale=processing_scale)
    
    return out

transfer_item_quantity = Node(transfer_item_quantity_setup,
                              transfer_item_quantity_exec)