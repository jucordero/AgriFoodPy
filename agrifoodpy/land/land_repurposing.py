import xarray as xr
import numpy as np

from agrifoodpy.pipeline import Node, standalone

def land_repurposing_setup(datablock):
    return datablock

@standalone(["land", "mask"], ["land"])
def land_repurposing_exec(land, land_type, fraction, new_types,
                          ratio=None, mask=None, mask_values=None,
                          datablock=None):
    """Replaces a fraction of an input list of land types by a new or existing
    type, assuming each pixel contains percentage data".

    Parameters
    ----------
    land : str
        Datablock path to the dataset containing the land use data.
    land_type : str, list
        The types of land to be replaced.
    fraction : float
        Fraction of the land type to be replaced.
    new_types : list
        List of new land types to be added to the land use dataset.
    ratio : float
        Relative proportion of the new land types to be added to the new land
        use type, with respect to the total repurposed land.
    """

    # Create datablock if it not provided
    standalone = False
    if datablock is None:
        datablock = {
            "land" : land,
        }
        land = "land"
        if mask is not None:
            datablock["mask"] = mask
            mask = "mask"
        standalone = True

    # Load land use data from datablock
    pctg = datablock[land].copy(deep=True)

    # Ensure land_type, new_types and ratio are arrays
    if np.isscalar(new_types):
        new_types = [new_types]

    ratio = np.array(ratio) if ratio is not None else np.ones(len(new_types))

    if np.isscalar(land_type):
        land_type = [land_type]

    # Check if new_types and ratio have the same number of elements
    if len(new_types) != len(ratio):
        raise ValueError("new_types and ratio must have the same number of elements")

    # Normalize ratio to sum to 1
    ratio = ratio / np.sum(ratio)

    # if no mask array is provided, then use the whole map
    for lt in land_type:

        if mask is not None:
            mask_arr = datablock[mask]
            mask_arr_data = np.isin(mask_arr, mask_values)
        else:
            mask_arr_data = np.ones_like(pctg, dtype=bool)

        to_repurpose = pctg.where(mask_arr_data, other=0).sel({"aggregate_class":lt})

        # Compute spared fraction to be re forested and remove from the spared class
        delta_repurposed = to_repurpose * fraction
        pctg.loc[{"aggregate_class":lt}] -= delta_repurposed

        for type, r in zip(new_types, ratio):
            if type not in pctg.aggregate_class.values:
                new_class = xr.zeros_like(pctg.isel(aggregate_class=0)).where(np.isfinite(pctg.isel(aggregate_class=0)))
                new_class["aggregate_class"] = type
                pctg = xr.concat([pctg, new_class], dim="aggregate_class")
            pctg.loc[{"aggregate_class":type}] += delta_repurposed * r

    # Rewrite land use data to datablock
    datablock[land] = pctg

    if standalone:
        return datablock[land]

    return datablock

land_repurposing = Node(land_repurposing_setup, land_repurposing_exec)