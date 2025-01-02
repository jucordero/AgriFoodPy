import numpy as np
import xarray as xr
from agrifoodpy.pipeline import Node, standalone

def quantities_per_capita_setup(datablock):
    return datablock

@standalone(input_keys=["fbs"], return_keys=["key"])
def quantities_per_capita_exec(fbs, population, key=None, datablock=None):
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

quantities_per_capita = Node(quantities_per_capita_setup,
                             quantities_per_capita_exec)