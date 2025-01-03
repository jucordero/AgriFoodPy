from agrifoodpy.pipeline import standalone
from agrifoodpy.pipeline.utils import item_parser
import xarray as xr
import numpy as np
import warnings

@standalone(["dataset"], ["dataset"])
def balanced_item_scaling(dataset, element, items, scale, source=None,
                               items_out=None, elasticity=None, constant=True,
                               timescale=None, start_year=None, adoption="logistic",
                               datablock=None, fallback=None, add_fallback=True,
                               scale_feed_seed=False):
    """
    Scales selected item quantities in a Food balance sheet while keeping the
    total sum in an array constant.
    """
    if isinstance(timescale, str):
        timescale = datablock[timescale]

    if isinstance(start_year, str):
        start_year = datablock[start_year]

    data = datablock[dataset].copy(deep=True)

    # If items are provided in tuple use labeling coordinate
    items = item_parser(data, items)
    items_out = item_parser(data, items_out)

    if np.isscalar(source):
        source = [source]

    # Check for single item list fbs
    input_item_list = data.Item.values
    if np.isscalar(input_item_list):
        input_item_list = [input_item_list]
        if constant:
            warnings.warn("Constant set to true but input only has a single item.")
            constant = False

    # If no items are provided, we scale all of them.
    if np.sort(items) is np.sort(input_item_list):
        items = data.Item.values
        if constant:
            warnings.warn("Cannot keep food constant when scaling all items.")
            constant = False

    # Define scale array
    if adoption == "linear":
        from agrifoodpy.utils.scaling import linear_scale as scale_func
    elif adoption == "logistic":
        from agrifoodpy.utils.scaling import logistic_scale as scale_func
    else:
        raise ValueError("Adoption must be one of 'linear' or 'logistic'")

    y0 = data.Year.values[0]
    y1 = start_year
    y2 = np.min([start_year + timescale, data.Year.values[-1]])
    y3 = data.Year.values[-1]
    
    scale_arr = scale_func(y0, y1, y2, y3, c_init=1, c_end = scale)

    # Scale and add to source element
    out = data.fbs.scale_add(element, source, scale_arr, items, add=True,
                             elasticity=elasticity)


    if constant:

        delta = out[element] - data[element]

        # Scale out items
        
        # If no items are provided, we scale all non-input items.
        if items_out is None:
            non_sel_items = np.setdiff1d(data.Item.values, items)

        else:
            # Check no items are repeated
            if any(item in items_out for item in items):
                raise ValueError("Items cannot be in both input and output \
                                 lists simultaneously")

            non_sel_items = items_out

        non_sel_scale = (data.sel(Item=non_sel_items)[element].sum(dim="Item") -
                         delta.sum(dim="Item")) / data.sel(Item=non_sel_items)[element].sum(dim="Item")
        
        # Make sure inf and nan values are not scaled
        non_sel_scale = non_sel_scale.where(np.isfinite(non_sel_scale)).fillna(1.0)

        if np.any(non_sel_scale < 0):
            warnings.warn("Additional consumption cannot be compensated by \
                        reduction of non-selected items")

        out = out.fbs.scale_add(element, source, non_sel_scale, non_sel_items,
                                add=True, elasticity=elasticity)
        
        # If fallback is defined, adjust to prevent negative values
        if fallback is not None:
            df = sum(out[org].where(out[org] < 0).fillna(0) for org in source)
            out[fallback] -= np.where(add_fallback, -1, 1)*df
            for org in source:
                out[org] = out[org].where(out[org] > 0, 0)

    if scale_feed_seed:
        out = _feed_seed_processing_scale(out, data,
                                          items_feed=("Item_origin", "Animal Products"),
                                          items_seed=("Item_origin", "Vegetal Products"),
                                          items_processing=out.Item.values)

    ratio = out / data
    ratio = ratio.where(~np.isnan(ratio), 1)

    datablock[dataset] *= ratio

    return datablock

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
