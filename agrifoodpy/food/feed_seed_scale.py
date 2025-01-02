

def feed_scale(fbs, ref, items):
    """Scales feed, seed and processing quantities according to the change
    in production of animal and vegetal products
    
    Parameters
    """

    feed_scale = fbs["production"].sel(Item=fbs.Item_origin=="Animal Products").sum(dim="Item") \
                / ref["production"].sel(Item=ref.Item_origin=="Animal Products").sum(dim="Item")

    seed_scale = fbs["production"].sel(Item=fbs.Item_origin=="Vegetal Products").sum(dim="Item") \
                / ref["production"].sel(Item=ref.Item_origin=="Vegetal Products").sum(dim="Item")
    
    processing_scale = fbs["production"].sum(dim="Item") \
                / ref["production"].sum(dim="Item")

    out = fbs.fbs.scale_add(element_in="feed", element_out="production",
                            scale=feed_scale)
    
    out = out.fbs.scale_add(element_in="seed",element_out="production",
                            scale=seed_scale)
    
    out = out.fbs.scale_add(element_in="processing",element_out="production",
                            scale=processing_scale)
    return out