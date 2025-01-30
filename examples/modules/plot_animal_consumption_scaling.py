"""
========================================
Reducing UK's animal product consumption
========================================


This example demonstrates the use of individual module functions, without the
need to create a pipeline.

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
from agrifoodpy.pipeline.utils import item_parser
from agrifoodpy.food.model import balanced_item_scaling

# Select food items and production values for the 90s and 00s decades of data
# in the UK. Values are in 1000 Tonnes

country_code = 229
years = np.arange(1990, 2010)
food_uk = load_dataset(module="agrifoodpy_data.food",
                       data_attr="FAOSTAT",
                       coords={"Year":years, "Region":country_code})

animal_items = item_parser(food_uk, ("Item_origin", "Animal Products"))

# Scale domestic use of animal items by a factor of 0.5, while keeping
# the sum of domestic use constant. Reduce imports to account for the new
# consumption values
food_uk_scaled = balanced_item_scaling(food_uk,
                                       element="domestic",
                                       items=animal_items,
                                       scale=0.5,
                                       timescale=10,
                                       start_year=1995,
                                       constant=True,
                                       source="production",
                                       fallback="imports")

# We group the original and scaled quantities by origin and plot to compare
food_uk_origin = food_uk.fbs.group_sum("Item_origin")
food_uk_scaled_origin = food_uk_scaled.fbs.group_sum("Item_origin")

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