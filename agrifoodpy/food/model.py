""" Module for food intervention models and interfaces with external packages
"""

import xarray as xr
import numpy as np
import copy
from agrifoodpy.pipeline import standalone
from agrifoodpy.pipeline.utils import item_parser
from agrifoodpy.utils.scaling import logistic_scale
import warnings

@standalone(input_keys=["fbs"], return_keys=["fbs"])
def reduce_excess(fbs, element, source, threshold, percentage=1.0,
                       timescale=None, start_year=None, fallback=None,
                       datablock=None):
    """Reduces a fraction of the sum of an food balance sheet element above a
    specified threshold.
    
    Parameters
    ----------
    fbs : str, xarray.Dataset
        Datablock key to the a Food balance sheet dataset, or the dataset itself
    element : str
        Element of the food balance sheet to be scaled.
    source : str,
        Element of the food balance sheet to be used as source of the modified
        element quantities.
    threshold : str, float
        Datablock key to the threshold value to be used, or the threshold value
        itself.
    percentage : float
        Percentage of the quantity above the threshold to be reduced in the food
        balance sheet.
        Optional scaling factor or array to convert quantities prior to scaling.
    timescale : int, optional
        Timescale for the scaling to be applied completely.
    start_year: int, optional
        Year of the Food Balance Sheet to use as starting point for the scaling.
        If "start_year" + "timescale" is greater than the last year in the
        array, the scaling is truncated to the last year in the array.
    datablock : xarray.Dataset
        Datablock containing the food balance sheet dataset.
    """

    # Retrieve values from datablock if a key is given
    if isinstance(timescale, str):
        timescale = datablock[timescale]

    if isinstance(start_year, str):
        start_year = datablock[start_year]

    if isinstance(threshold, str):
        threshold = datablock[threshold]
    
    food = copy.deepcopy(datablock[fbs])

    # Maximum excess fractional reduction
    max_factor = (food[element].isel(Year=-1).sum(dim="Item") - threshold) \
                 / food[element].isel(Year=-1).sum(dim="Item") \
                 * percentage
    
    max_factor = max_factor.to_numpy()

    # Create a logistic curve starting at 1, ending at 1-max_factor
    y0 = food.Year.values[0]
    y1 = start_year
    y2 = np.min([start_year + timescale, food.Year.values[-1]])
    y3 = food.Year.values[-1]

    scale_waste = logistic_scale(y0, y1, y2, y3, c_init=1, c_end=1-max_factor)

    # Scale food and subtract difference from source element
    out = food.fbs.scale_add(element_in=element,
                             element_out=source,
                             scale=scale_waste)
    
    # If supply element is negative, set to zero and add the negative delta to imports
    out = _check_negative_source(out, source, fallback)

    datablock[fbs] = out

    return datablock

@standalone(input_keys=["dataset"], return_keys=["dataset"])
def scale_add_items(dataset, in_array, out_array, items, scale,
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

    datablock[dataset] = out

    return datablock

@standalone(input_keys=["fbs"], return_keys=["key"])
def quantities_per_capita(fbs, population, key=None, datablock=None):
    """Converts a food balance sheet into per capita quantities.
    
    Parameters
    ----------
    datablock : Dict
        Dictionary containing data.
    fbs : str, xarray.Dataset
        Datablock path to the food balance sheet dataset or the dataset itself.
    population : str, xarray.Dataarray
        Datablock path to the dataset containing population data, or the arrray
        itself.
    key : str
        Key of the resulting dataset to be stored in the datablock.

    Returns
    -------
    dict or xarray.Dataset
        - Updated datablock if  a datablock is provided.
        - xarray.Dataset with per capita quantities if no datablock is provided.
    """

    # If no optional key is given, use the name of the fbs dataset
    if key is None:
        key = fbs

    pop = datablock[population]
    fbs = datablock[fbs]

    # extract population for the year range of the food balance sheet
    pop_fbs = pop.sel(Year=fbs.Year.values)

    # compute per capita quantities
    food_cap_day = fbs/pop_fbs/365.25

    datablock[key] = food_cap_day

    return datablock

@standalone(input_keys=["dataset"], return_keys=["dataset"])
def transfer_item_quantity(dataset, element, item, item_out, scale, source,
                                fallback=None, timescale=None, start_year=None,
                                datablock=None):
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
    
    # If production is negative, set to zero and add the negative delta to
    # imports
    if fallback is not None:
        out = _check_negative_source(out, source, fallback)
    
    # Add delta to cultured meat
    delta = (food_orig-out).sel(Item=items_to_replace).expand_dims("Item").sum(dim="Item")
    out.loc[{"Item":item_out}] += delta
    datablock[dataset] = out

    return datablock

def _check_negative_source(fbs, source, fallback):
    """Checks for negative values in the source element and adds the difference
    to the fallback element"""

    delta_neg = fbs[source].where(fbs[source] < 0, other=0)
    fbs[source] -= delta_neg
    fbs[fallback] += delta_neg

    return fbs

@standalone(input_keys=["fbs", "ref"], return_keys=["fbs"])
def feed_seed_scale(fbs, ref, feed_items, seed_items, production="production",seed="seed",
               feed="feed", processing="processing", datablock=None):
    """Scales feed, seed and processing quantities according to the change
    in production of items requiring them.
    
    Parameters
    ----------
    fbs : xarray.Dataset
        FBS dataset
    ref : xarray.Dataset
        Reference FBS dataset
    feed_items : list
        List of items requiring feed
    seed_items : list
        List of items requiring seed
    production : str
        Name of the production element
    seed : str
        Name of the seed element
    feed : str
        Name of the feed element
    processing : str
        Name of the processing element
    datablock : dict
        Datablock dictionary
    """

    ref_dataset = datablock[ref]
    food_orig = datablock[fbs]
    
    feed_items = item_parser(food_orig, feed_items)
    seed_items = item_parser(food_orig, seed_items)
    
    feed_scale = food_orig[production].sel(Item=feed_items).sum(dim="Item") \
                / ref_dataset[production].sel(Item=feed_items).sum(dim="Item")

    seed_scale = food_orig[production].sel(Item=seed_items).sum(dim="Item") \
                / ref_dataset[production].sel(Item=seed_items).sum(dim="Item")
    
    processing_scale = food_orig[production].sum(dim="Item") \
                / ref_dataset[production].sum(dim="Item")

    out = food_orig.fbs.scale_add(element_in=feed, element_out=production,
                            scale=feed_scale)
    
    out = out.fbs.scale_add(element_in=seed, element_out=production,
                            scale=seed_scale)
    
    out = out.fbs.scale_add(element_in=processing ,element_out=production,
                            scale=processing_scale)
    
    datablock[fbs] = out

    return datablock

@standalone(["dataset"], ["dataset"])
def balanced_item_scaling(dataset, element, items, scale, source=None,
                          items_out=None, elasticity=None, constant=True,
                          timescale=None, start_year=None, adoption="logistic",
                          datablock=None, fallback=None, add_fallback=True):
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

    datablock[dataset] = out

    return datablock

@standalone(input_keys=["fbs"], return_keys=["fbs"])
def fbs_convert(fbs, convertion_arr, keys=None, datablock=None):
    """Converts quantities in the food balance sheet using a conversion
    dataarray, dataset, or scaling factor.
    
    Parameters
    ----------
    datablock : Dict
        Dictionary containing data.
    dataset : str, xarray.Dataset
        Datablock paths to the food balance sheet datasets or the datasets
        themselves.
    convertion_arr : str, xarray.DataArray, tuple
        Datablock path to the conversion array, dataset-key tuple, or the array
        itself.
    keys : str, list
        Datablock key of the resulting dataset to be stored in the datablock.

    Returns
    -------
    dict or xarray.Dataset
        - Updated datablock if  a datablock is provided.
        - xarray.Dataset with converted quantities if no datablock is provided.
    """

    data = datablock[fbs]    

    # Prepare convertion array
    if isinstance(convertion_arr, str):
        convertion_arr = datablock[convertion_arr]
    
    elif isinstance(convertion_arr, tuple):
        convertion_arr = datablock[convertion_arr[0]][convertion_arr[1]]

    if isinstance(convertion_arr, xr.DataArray):
        convertion_arr = convertion_arr.where(np.isfinite(convertion_arr), other=0)

    # If no key is provided, overwrite original dataset
    if keys is None:
        keys = fbs

    datablock[keys] = data*convertion_arr

    return datablock

def balanced_scaling(fbs, items, scale, element, year=None, adoption=None, 
                     timescale=10, origin=None, constant=False,
                     fallback=None):
    """Scale items quantities across multiple elements in a FoodBalanceSheet
    Dataset 
    
    Scales selected item quantities on a food balance sheet and with the
    posibility to keep the sum of selected elements constant.
    Optionally, produce an Dataset with a sequence of quantities over the years
    following a smooth scaling according to the selected functional form.

    The elements used to supply the modified quantities can be selected to keep
    a balanced food balance sheet.

    Parameters
    ----------
    fbs : xarray.Dataset
        Input food balance sheet Dataset.
    items : list
        List of items to scale in the food balance sheet.
    element : string
        Name of the DataArray to scale.
    scale : float
        Scaling parameter after full adoption.
    year : int, optional
        Year of the Food Balance Sheet to use. If not set, the last year of the 
        array is used
    adoption : string, optional
        Shape of the scaling adoption curve. "logistic" uses a logistic model
        for a slow-fast-slow adoption. "linear" uses a constant slope adoption
        during the the "timescale period"
    timescale : int, optional
        Timescale for the scaling to be applied completely.  If "year" +
        "timescale" is greater than the last year in the array, it is extended
        to accomodate the extra years.
    origin : string, optional
        Name of the DataArray which will be used to balance the food balance
        sheets. Any change to the "element" DataArray will be reflected in this
        DataArray.
    constant : bool, optional
        If set to True, the sum of element remains constant by scaling the non
        selected items accordingly.
    fallback : string, optional
        Name of the DataArray used to provide the excess required to balance the
        food balance sheet in case the "origin" falls below zero.

    Returns
    -------
    data : xarray.Dataarray
        Food balance sheet Dataset with scaled "food" values.
    """

    # Check for single item inputs
    if np.isscalar(items):
        items = [items]

    # Check for single item list fbs
    input_item_list = fbs.Item.values
    if np.isscalar(input_item_list):
        input_item_list = [input_item_list]
        if constant:
            warnings.warn("Constant set to true but input only has a single item.")
            constant = False

    # If no items are provided, we scale all of them.
    if items is None or np.sort(items) is np.sort(input_item_list):
        items = fbs.Item.values
        if constant:
            warnings.warn("Cannot keep food constant when scaling all items.")
            constant = False

    # Define Dataarray to use as pivot
    if "Year" in fbs.dims:
        if year is None:
            if np.isscalar(fbs.Year.values):
                year = fbs.Year.values
                fbs_toscale = fbs
            else:
                year = fbs.Year.values[-1]
                fbs_toscale = fbs.isel(Year=-1)
        else:
            fbs_toscale = fbs.sel(Year=year)

    else:
        fbs_toscale = fbs
        try:
            year = fbs.Year.values
        except AttributeError:
            year=0

    # Define scale array based on year range
    if adoption is not None:
        if adoption == "linear":
            from agrifoodpy.utils.scaling import linear_scale as scale_func
        elif adoption == "logistic":
            from agrifoodpy.utils.scaling import logistic_scale as scale_func
        else:
            raise ValueError("Adoption must be one of 'linear' or 'logistic'")
        
        scale_arr = scale_func(year, year, year+timescale-1, year+timescale-1,
                               c_init=1, c_end = scale)
        
        fbs_toscale = fbs_toscale * xr.ones_like(scale_arr)
    
    else:
        scale_arr = scale

    # Create a deep copy to modify and return
    out = fbs_toscale.copy(deep=True)
    osplit = origin.split("-")[-1]
    
    out = out.fbs.scale_add(element, osplit, scale_arr, items, 
                            add = origin.startswith("-"))
    

    if constant:

        delta = out[element] - fbs_toscale[element]

        # Scale non selected items
        non_sel_items = np.setdiff1d(fbs_toscale.Item.values, items)
        non_sel_scale = (fbs_toscale.sel(Item=non_sel_items)[element].sum(dim="Item") - delta.sum(dim="Item")) / fbs_toscale.sel(Item=non_sel_items)[element].sum(dim="Item")
        
        # Make sure inf and nan values are not scaled
        non_sel_scale = non_sel_scale.where(np.isfinite(non_sel_scale)).fillna(1.0)

        if np.any(non_sel_scale < 0):
            warnings.warn("Additional consumption cannot be compensated by \
                        reduction of non-selected items")
        
        out = out.fbs.scale_add(element, osplit, non_sel_scale,
                        non_sel_items, add = origin.startswith("-"))

        # If fallback is defined, adjust to prevent negative values
        if fallback is not None:
            df = out[osplit].where(out[osplit] < 0).fillna(0)
            out[fallback.split("-")[-1]] -= np.where(fallback.startswith("-"), 1, -1)*df
            out[osplit] = out[osplit].where(out[osplit] > 0, 0)

    return out