from agrifoodpy.pipeline import Node
import copy

def copy_datablock_setup(datablock):
    return datablock

def copy_datablock_exec(datablock, key, out_key):
    """Copy a datablock element into a new key in the datablock
    
    Parameters
    ----------
    datablock : xarray.Dataset
        The datablock to print
    key : str
        The key of the datablock to print
    out_key : str
        The key of the datablock to copy to

    Returns
    -------
    datablock : dict
        Datablock to with added key
    """

    datablock[out_key] = copy.deepcopy(datablock[key])

    return datablock

copy_datablock = Node(copy_datablock_setup, copy_datablock_exec)