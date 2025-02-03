"""
========================================
Reducing UK's animal product consumption
========================================

This example demonstrates the use of the pipeline manager to create a simple
pipeline of modules.

In this particular example, a food supply array is combined with a scaling
model to reduce animal product consumption.
It also employs a few functions from the ``fbs`` accessor to group and plot
food balance sheet arrays.

Consumption of animal based products is halved, while keeping total comsumed
weight constant by upscaling the consumption of vegetal products.
"""

import numpy as np
from matplotlib import pyplot as plt

from agrifoodpy.utils.load_dataset import load_dataset
from agrifoodpy.food.model import balanced_item_scaling
from agrifoodpy.utils.copy_datablock import copy_datablock
from agrifoodpy.pipeline import Pipeline

# Create a pipeline object
fs = Pipeline()

# Define the parameters for the pipeline nodes to be executed

# Load the food balance sheet using the load dataset module, selecting the UK
# region and years 1990 to 2010
load_food_params = {
    "datablock_path": "food",
    "module": "agrifoodpy_data.food",
    "data_attr": "FAOSTAT",
    "coords": {"Year":np.arange(1990, 2010), "Region":229}
}

# Add the node to the pipeline
fs.add_node(load_dataset, load_food_params)

# Create a copy of the "food" datablock element to use as a baseline
copy_datablock_params = {
    "key" : "food",
    "out_key" : "food_baseline"
}

fs.add_node(copy_datablock, copy_datablock_params)

# Define the parameters for the scaling module, which will reduce the
# domestic use of animal products by half. The remaining weight is taken from
# the "imports" element. We also set the "constant" parameter to True, so the
# total weight consumed remains constant.
scaling_params = {
    "dataset" : "food",
    "element" : "domestic",
    "items" : ("Item_origin", "Animal Products"),
    "scale" : 0.5,
    "timescale" : 10,
    "start_year" : 1995,
    "constant" : True,
    "source" : "production",
    "fallback" : "imports"}

fs.add_node(balanced_item_scaling, scaling_params)

# Run the pipeline
fs.run()

results = fs.datablock

# We group the original and scaled quantities by origin and plot to compare
food_uk_origin = results["food_baseline"].fbs.group_sum("Item_origin")
food_uk_scaled_origin = results["food"].fbs.group_sum("Item_origin")

#%%
# From the plot we can see that domestic use of animal products is reduced by
# half, while keeping total weight constant. We used ``-exports`` as the
# fallback for any extra origin required. If any item domestic use reduction
# requires more origin reduction than available, the remaining is taken from
# the ``fallback`` DataArray element.

# Plot and compare values before and after
f, axes = plt.subplots(2,1, sharex=True)
plt.subplots_adjust(hspace=0)

food_uk_origin.isel(Year=-1).fbs.plot_bars(show="Item_origin",
                                  elements=["production", "imports"],
                                  inverted_elements=["exports", "domestic"],
                                  labels="show", ax=axes[0])

food_uk_scaled_origin.isel(Year=-1).fbs.plot_bars(show="Item_origin",
                                  elements=["production", "imports"],
                                  inverted_elements=["exports", "domestic"],
                                  labels="show", ax=axes[1])

axes[1].set_xlabel("1000 Tonnes")
plt.tight_layout()

plt.show()