""" Module for impact intervention models
"""
from agrifoodpy.pipeline import standalone

import xarray as xr
import numpy as np

def fbs_impacts(fbs, impact_element, population=None, sum_dims=None):
    """Computes total impacts from quantities in a food balance sheet Dataset or
    food element quantity DataArray, summing over items, regions or years if
    instructed.

    Parameters
    ----------
    fbs : xarray.DataArray or xarray.Dataset
        Food Balance Sheet array with food quantities for a set of items,
        regions and/or years.         
    impact_element : xarray.DataArray
        Impact DataArray containing the impacts for set of items, regions and/or
        years.
    population : xarray.DataArray
        If given, the input impacts are considered per-capita values and
        multiplied by the population array

    sum_dims : str
        Dimension labels to sum over
    
    Returns
    -------
    total_impact : xarray.DataArray or xarray.Dataset
        Total impact computed from food balance sheet data and impact array
    """

    total_impact = fbs * impact_element
    if population is not None:
        total_impact *= population

    if sum_dims is not None:
        total_impact = total_impact.sum(dim=sum_dims)
        
    return total_impact

def fair_co2_only(emissions):
    """Simple Interface to FaIR, the Finite-amplitude Impulse-Response
    atmosferic model.

    Computes the concentration, radiative forcing and temperature anomaly for an 
    array of CO2 emissions per year assuming a clean atmosphere and default
    values for amosferic parameters.

    Parameters
    ----------
    emissions : xarray.DataArray or xarray.Dataset
        Array containing GHG emissions in Gt CO2e per year 

    Returns
    -------
    T : xarray.DataArray
        Temperature anomaly in Kelvin degrees at the zero layer
    C : xarray.DataArray
        Atmosferic CO2e concetration in ppm 
    F : xarray.DataArray
        Effective radiative forcing in W m^-2
    """

    from fair import FAIR
    from fair.interface import fill, initialise
    f = FAIR()

    years = np.unique(emissions.Year.values)

    # Configure method, timebounds, and labels
    f.ghg_method='myhre1998'
    f.define_time(years[0]-0.5, years[-1]+0.5, 1)
    f.define_scenarios(["default"])
    f.define_configs(["default"])

    # Define CO2 as the only specie
    species = ['CO2']
    properties = {
        'CO2': {
            'type': 'co2',
            'input_mode': 'emissions',
            # it doesn't behave as a GHG itself in the model, but as a precursor
            'greenhouse_gas': True,  
            'aerosol_chemistry_from_emissions': False,
            'aerosol_chemistry_from_concentration': False,
        }}
    
    f.define_species(species, properties)

    # Allocate arrays
    f.allocate()

    # Set default values
    fill(f.climate_configs["ocean_heat_transfer"], [1.1, 1.6, 0.9],
         config='default')
    
    fill(f.climate_configs["ocean_heat_capacity"], [8, 14, 100],
         config='default')
    
    fill(f.climate_configs["deep_ocean_efficacy"], 1.1, config='default')

    # Set initial conditions.
    initialise(f.concentration, 278.3, specie='CO2')
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # Fill species configs
    f.fill_species_configs()
    
    f.emissions.loc[{"scenario":"default",
                     "specie":"CO2",
                     "config":"default"}] = emissions.to_numpy()
    
    # Run and return
    f.run(progress=False)

    return_dict = {"scenario":"default", "config":"default"}

    T = f.temperature.sel(return_dict).drop_vars(
        ["scenario", "config", "layer"]).squeeze()
    
    C = f.concentration.sel(return_dict).drop_vars(
        ["scenario", "config", "specie"]).squeeze()
    
    F = f.forcing.sel(return_dict).drop_vars(
        ["scenario", "config", "specie"]).squeeze()

    return T.sel(layer=0), C, F

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


@standalone([], ["output"])
def add_sequestration(type, max_seq, years, output="output", 
                      start_year=None, timescale=None,
                      scale_func="logistic", datablock=None):
    """Adds total annual sequestration from different sources according to a 
    maximum annual sequestration and a scaling function.

    Parameters
    ----------
    seq : str
        Datablock key for the sequestration dataset.
    type : str
        Sequestration type to be used as DataArray name.
    max_seq : float
        The maximum annual sequestration in t CO2e/year.
    years : array, xarray.DataArray, xarray.DataSet, str
        Year array to use for the output dataset. If a datablock key is given,
        the year range of the dataset at the given key is used. If an xarray
        DataArray or xarray.DataSet is given, their year range is used.
    sequestration : str
        Datablock key to store the sequestration dataset.
    start_year : int
        Year to start the computation.
    timescale : int
        Timescale to use for the adoption scaling function.
    scale_func : str
        Scaling function to use for the adoption scaling. Must be either
        'logistic' or 'linear'
    datablock : dict
        The datablock dictionary.
    """

    if isinstance(timescale, str):
        timescale = datablock[timescale]

    # Make sue the type and max_seq are lists
    if np.isscalar(type):
        type = [type]

    if np.isscalar(max_seq):
        max_seq = [max_seq]

    # Load scaling function
    if scale_func == "logistic":
        from agrifoodpy.utils.scaling import logistic_scale as scale_function
    elif scale_func == "linear":
        from agrifoodpy.utils.scaling import linear_scale as scale_function
    else:
        raise ValueError("Scale must be either 'logistic' or 'linear'")
    
    if isinstance(years, str):
        years = datablock[years].Year.values
    elif isinstance(years, [xr.DataArray, xr.Dataset]):
        years = years.Year.values
    
    scale = scale_function(years[0], start_year, start_year+timescale,
                                        years[-1], c_init=0, c_end=1)
    
    # Write sequestration to datablock
    for type, seq in zip(type, max_seq):

        seq_arr = seq * scale

        # Create a dataset with the different sequestration sources
        seq_ds = xr.Dataset({type: seq_arr})
    
        seq_da = seq_ds.to_array(dim="Item", name=output)
    
        if output not in datablock:
            datablock[output] = seq_da
        else:
            # append sequestration to existing sequestration da
            seq_da_in = datablock[output]
            seq_da = xr.concat([seq_da_in, seq_da], dim="Item")
            datablock[output] = seq_da

    return datablock
