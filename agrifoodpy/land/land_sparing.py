import xarray as xr
import numpy as np

def land_sparing(datablock, spare_fraction, land, land_type,
                      spare_label='Spared', mask=None, mask_values=None,
                      food=None, items=None, timescale=None, year_start=None,
                      adoption='logistic'):
    """Replaces a specified land type fraction and sets it to a new type called
    given by spared_label. If a food balance sheet is provided, it also 
    scales food production and imports to reflect the change in land use.

    Parameters
    ----------
    spare_fraction : float
        Fraction of the land type to spare.
    land : str
        Datablock key for the land use dataset. The model assumes a land
        percentage dataset with percentages for each class on each percentage.
    land_type : str
        The land types that will be spared.
    mask : str
        Datablock key for the mask dataset used to select specific locations of
        the land dataset.
    mask_values : list
        List of values in the mask dataset to use for masking.
    food : str
        Datablock key for the food balance sheet dataset.
    items : list
        List of items to be scaled in the food balance sheet
    timescale : int, str
        Datablock key for the timescale dataset, or the timescale in years
        itself.
    year_start : int
        Year to start the scaling from.
    adopton : str
        Type of adoption curve to use. Can be 'logistic' or 'linear'.
    """
    
    pctg = datablock[land].copy(deep=True)

    old_use = datablock[land].sel({"aggregate_class":land_type}).sum()

    # if no mask array is provided, then use the whole map
    if mask is not None:
        alc = datablock[mask]
        alc_mask = np.isin(alc, mask_values)
    else:
        alc_mask = np.ones_like(pctg, dtype=bool)

    to_spare = pctg.where(alc_mask, other=0).sel({"aggregate_class":land_type})

    # Spare the specified land type
    delta_spared =  to_spare * spare_fraction
    pctg.loc[{"aggregate_class":land_type}] -= delta_spared

    if "Spared" not in pctg.aggregate_class.values:
        spared_new_class = xr.zeros_like(pctg.isel(aggregate_class=0)).where(np.isfinite(pctg.isel(aggregate_class=0)))
        spared_new_class["aggregate_class"] = spare_label
        pctg = xr.concat([pctg, spared_new_class], dim="aggregate_class")

    pctg.loc[{"aggregate_class":spare_label}] += delta_spared.sum(dim="aggregate_class")

    # Add spared class to the land use map
    datablock[land] = pctg

    # Scale food production and imports if food array info is provided
    if food is not None:

        food_orig = datablock[food]
        
        if isinstance(timescale, str):
            timescale = datablock[timescale]

        # Scale food production and imports
        new_use = pctg.sel({"aggregate_class":land_type}).sum()
        scale_use = (new_use/old_use).to_numpy()

        # Define scale array based on year range
        if adoption is not None:
            if adoption == "linear":
                from agrifoodpy.utils.scaling import linear_scale as scale_func
            elif adoption == "logistic":
                from agrifoodpy.utils.scaling import logistic_scale as scale_func
            else:
                raise ValueError("Adoption must be one of 'linear' or 'logistic'")
            
        y0 = food_orig.Year.values[0]
        y1 = year_start
        y2 = np.min([year_start + timescale, food_orig.Year.values[-1]])
        y3 = food_orig.Year.values[-1]
            
        scale_spare = scale_func(y0, y1, y2, y3, c_init=1, c_end=scale_use)

        scaled_items = food_orig.sel(Item=food_orig.Item_origin==items).Item.values

        out = food_orig.fbs.scale_add(element_in="production",
                                    element_out="imports",
                                    scale=scale_spare,
                                    items=scaled_items,
                                    add=False)
        
        ratio = out / food_orig
        ratio = ratio.where(~np.isnan(ratio), 1)

        # Update per cap/day values and per year values using the same ratio, which
        # is independent of population growth
        qty_key = ["g/cap/day", "g_prot/cap/day", "g_fat/cap/day", "kCal/cap/day"]
        for key in qty_key:
            datablock[key] *= ratio

    # datablock["food"]["g/cap/day"] = out

    return datablock
