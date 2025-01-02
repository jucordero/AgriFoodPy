from agrifoodpy.pipeline import Node

def print_datablock_setup(datablock):
    return datablock

def print_datablock_exec(datablock, key, sel={}):
    """Prints a datablock element at any point in the pipeline execution
    
    Parameters
    ----------
    datablock : xarray.Dataset
        The datablock to print
    key : str
        The key of the datablock to print
    sel : dict, optional
        The selection to apply to the datablock

    Returns
    -------
    datablock : xarray.Dataset
        Unmodified datablock to continue execution
    """

    print(datablock[key].sel(sel))

    return datablock

print_datablock = Node(print_datablock_setup, print_datablock_exec)