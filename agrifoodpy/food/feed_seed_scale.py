from agrifoodpy.pipeline import standalone
from agrifoodpy.pipeline.utils import item_parser

@standalone(input_keys=["fbs", "ref"], return_keys=["fbs"])
def feed_seed_scale(fbs, ref, feed_items, seed_items, production="production",seed="seed",
               feed="feed", processing="processing", datablock=None):
    """Scales feed, seed and processing quantities according to the change
    in production of items requiring them.
    
    Parameters
    ----------
    fbs : xarray.Dataset
        FBS dataset
    ref : xarray.Dataset
        Reference FBS dataset
    feed_items : list
        List of items requiring feed
    seed_items : list
        List of items requiring seed
    production : str
        Name of the production element
    seed : str
        Name of the seed element
    feed : str
        Name of the feed element
    processing : str
        Name of the processing element
    datablock : dict
        Datablock dictionary
    """

    ref_dataset = datablock[ref]
    food_orig = datablock[fbs]
    
    feed_items = item_parser(food_orig, feed_items)
    seed_items = item_parser(food_orig, seed_items)
    
    feed_scale = food_orig[production].sel(Item=feed_items).sum(dim="Item") \
                / ref_dataset[production].sel(Item=feed_items).sum(dim="Item")

    seed_scale = food_orig[production].sel(Item=seed_items).sum(dim="Item") \
                / ref_dataset[production].sel(Item=seed_items).sum(dim="Item")
    
    processing_scale = food_orig[production].sum(dim="Item") \
                / ref_dataset[production].sum(dim="Item")

    out = food_orig.fbs.scale_add(element_in=feed, element_out=production,
                            scale=feed_scale)
    
    out = out.fbs.scale_add(element_in=seed, element_out=production,
                            scale=seed_scale)
    
    out = out.fbs.scale_add(element_in=processing ,element_out=production,
                            scale=processing_scale)
    
    datablock[fbs] = out

    return datablock