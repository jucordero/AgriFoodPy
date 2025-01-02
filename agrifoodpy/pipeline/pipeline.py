"""Pipeline implementation

This class provides methods to build and manage a pipeline for end to end
simulations using the agrifoodpy package.
"""

from .node import Node

import copy
from functools import wraps
from inspect import signature

class Pipeline():
    def __init__(self):
        self.nodes = []
        self.setup_params = []
        self.exec_params = []
        self.datablock = {}

    def read(cls, filename):
        """Read a pipeline from a configuration file

        Parameters
        ----------
        filename : str
            The name of the configuration file.

        Returns
        -------
        pipeline : Pipeline
            The pipeline object.
        """
        pass

    def add_node(self, node, setup_params={}, exec_params={}):
        """Adds a step to the pipeline, including its setup and execution
        functions.

        Parameters
        ----------
        setup_func : function
            The function to be called at the setup stage of the pipeline.

        setup_params : dict, optional
            The parameters to be passed to the setup stage function.

        exec_params : dict, optional
            The parameters to be passed to the execution stage function.
        """

        # Copy the parameters to avoid modifying the original dictionaries
        setup_params = copy.deepcopy(setup_params)
        exec_params = copy.deepcopy(exec_params)

        self.nodes.append(node)
        self.setup_params.append(setup_params)
        self.exec_params.append(exec_params)

    def run(self):
        """Runs the pipeline
        """
        self.run_setup()
        self.run_exec()

    def run_setup(self):
        """Runs the setup functions for each node
        """
        # Execute the setup functions for each node
        for node, setup_params in zip(self.nodes, self.setup_params):
            self.datablock = node.setup(datablock = self.datablock, **setup_params)

    def run_exec(self):
        """Runs the execution functions for each node
        """
        # Execute the execution functions for each node
        for node, exec_params in zip(self.nodes, self.exec_params):
            self.datablock = node.execute(datablock = self.datablock, **exec_params)


def standalone(input_keys, return_keys):
    """ Decorator to make a pipeline node available as a standalone function

    If datablock is not passed as a kwarg, and datasets are passed directly
    instead of datablock keys, a temporary datablock is created and the datasets
    associated with the arguments in input_keys are added to it. The function
    then returns the specified datasets in return_keys.

    Parameters
    ----------
    key_list: list of strings
        List of dataset keys to be added to the temporary datablock
    return_list: list of strings
        List of keys to datablock datasets to be returned by the decorated
        function.

    Returns
    -------

    wrapper: function
        The decorated function
    
    """
    def pipeline_decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):

            # Identify positional arguments
            func_sig = signature(test_func)
            func_params = func_sig.parameters

            kwargs.update({key: arg for key, arg in zip(func_params.keys(), args)})

            # Make sure that the datablock is passed as a kwarg, if not, create it
            datablock = kwargs.get("datablock", None)

            standalone = datablock is None
            if standalone:
                # Create datablock
                datablock = {key: kwargs[key] for key in kwargs if key in input_keys}
                kwargs["datablock"] = datablock
                
                # Create list of keys for passed arguments only
                for key in input_keys:
                    if kwargs.get(key, None) is not None:
                        kwargs[key] = key
            
            result = test_func(**kwargs)

            # return tuple of results
            if standalone:
                if len(return_keys) == 1:
                    return result[return_keys[0]]
                else:
                    return tuple(result[key] for key in return_keys)

            return result
        return wrapper
    return pipeline_decorator