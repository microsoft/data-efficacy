try:
    from .common import cross_guidance_order, sort_data, window_based_shuffle
except ImportError:
    from common import cross_guidance_order, sort_data, window_based_shuffle


def order(in_data, args):
    score_field = args.score_field
    ascending = getattr(args, "ascending", True)
    num_sections = args.num_sections
    folding_ratio = args.folding_ratio
    folding_layer = args.folding_layer
    window_size = getattr(args, "window_size", 0)
    seed = getattr(args, "seed", 42)

    sorted_data = sort_data(in_data, score_field, ascending=ascending)
    out_data = cross_guidance_order(
        sorted_data,
        num_sections=num_sections,
        transition_ratio=folding_ratio,
        folding_layer=folding_layer,
        mode="zigzag",
    )
    return window_based_shuffle(out_data, window_size=window_size, seed=seed)
