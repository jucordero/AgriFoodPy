import xarray as xr
import numpy as np
import copy
from agrifoodpy.pipeline import standalone
from agrifoodpy.utils.scaling import logistic_scale

@standalone(input_keys=["fbs"], return_keys=["fbs"])
def reduce_excess(fbs, element, source, threshold, percentage=1.0,
                       timescale=None, start_year=None, datablock=None):
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
    out = check_negative_source(out, source)

    datablock[fbs] = out

    return datablock

def check_negative_source(fbs, source):
    """Checks for negative values in the source element and adds the difference
    to the fallback element"""

    if source == "production":
        fallback = "imports"
    elif source == "imports":
        fallback = "production"
    elif source == "exports":
        fallback = "production"

    delta_neg = fbs[source].where(fbs[source] < 0, other=0)
    fbs[source] -= delta_neg
    fbs[fallback] += delta_neg

    return fbs