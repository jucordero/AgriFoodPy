"""Node class for the agrifoodpy pipeline.

This class provides methods to build a node for the agrifoodpy pipeline.
"""

class Node:
    def __init__(self, setup_func=None, exec_func=None):
        self.setup_func = setup_func
        self.exec_func = exec_func

    def setup(self, datablock, **params):
        """
        Executes the setup function for the node, if it exists, and returns the
        updated datablock.
        
        Paarameters
        -----------
        datablock: dict
            The input datablock to be processed.
        **params: dict
            Additional parameters to be passed to the setup function.
        
        Returns:
        --------
        datablock: dict
            The updated datablock after executing the setup function.
        """
                
        if self.setup_func:
            datablock = self.setup_func(datablock = datablock, **params)
        return datablock

    def execute(self, datablock, **params):
        """
        Executes the execution function for the node, if it exists, and returns
        the updated datablock.
        Parameters
        ----------
        datablock: dict
            The input datablock to be processed.
        **params: dict
            Additional parameters to be passed to the execution function.
        
        Returns
        -------
        datablock: dict
            The updated datablock after executing the execution function.
        """
        if self.exec_func:
            datablock = self.exec_func(datablock = datablock, **params)
        return datablock