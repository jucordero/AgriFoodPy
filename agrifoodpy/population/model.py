import numpy as np
import xarray as xr
from agrifoodpy.pipeline import standalone

@standalone(["dataset", "population"], ["dataset"])
def population_projection(dataset, population, food="food",
                               production="production", imports="imports",
                               exports="exports", per_capita=False,
                               datablock=None):
    
    """Projects a food balance sheet into the future using a population
    dataset.
    
    Parameters
    ----------
    datablock : Dict
        Dictionary containing data.
    dataset : str, xarray.DataArray, xarray.Dataset
        Datasets to project, or datablock keys to the datasets to scale.
    population : str, xarray.DataArray
        Population dataset to use to scale quantities, or datablock key to the
        population dataset used to scale quantities. Year range must exceed the 
        year range of the food balance sheet to project into the future.
    poduction : str
        Array name of the food balance sheet dataset containing production data.
        Total production quantity will remain constant as a function of time.
    imports : str
        Array name of the food balance sheet dataset containing imports data.
        Imports data will scale with population to account for extra domestic
        use from population growth.
    exports : str
        Array name of the food balance sheet dataset containing exports data.
        Exports data will remain constant as a function of time.
    per_capita : bool
        If True, data quantities are assumed to be per capita, hence an inverse
        scaling factor is applied to the population dataset.

    Returns
    -------
    dict or xarray.Dataset
        - If no datablock is provided, returns the xarray.Dataset with projected
          quantities.
        - If a datablock is provided, returns the datablock with the scaled
          element array on thei corresponding key.

    """

    pop = datablock[population]
    fbs = datablock[dataset]

    # Define past and future years from input arrays
    years_past = fbs.Year.values
    years_future = pop.Year.values[pop.Year.values > years_past[-1]]

    # Generate population projection using last present year as pivot
    proj_pop = pop.sel(Year=years_future) / pop.sel(Year=years_past[-1])
    scale_past = xr.DataArray(np.ones(len(years_past)), dims=["Year"],
                            coords={"Year": years_past})
    
    scale_pop = xr.concat([scale_past, proj_pop], dim="Year")

    # Add years to food balance sheet in constant mode
    fbs = fbs.fbs.add_years(years_future, "constant")

    if per_capita:
        # per capita production varies supplied by imports
        fbs = fbs.fbs.scale_add(element_in=production,
                                    element_out=imports,
                                    scale=1/scale_pop, add=False)
    
        # per capita exports varies supplied by imports
        fbs = fbs.fbs.scale_add(element_in=exports,
                                element_out=imports,
                                scale=1/scale_pop)
    else:
        # total food varies, supplied by imports
        fbs = fbs.fbs.scale_add(element_in=food,
                                element_out=imports,
                                scale=scale_pop)
    
    # Write to datablock
    datablock[dataset] = fbs

    return datablock
