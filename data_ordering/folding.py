try:
    from .common import folding_order, sort_data, window_based_shuffle
except ImportError:
    from common import folding_order, sort_data, window_based_shuffle


def order(in_data, args):
    score_field = args.score_field
    folding_layer = args.folding_layer
    ascending = getattr(args, "ascending", True)
    window_size = getattr(args, "window_size", 0)
    seed = getattr(args, "seed", 42)

    sorted_data = sort_data(in_data, score_field, ascending=ascending)
    out_data = folding_order(sorted_data, folding_layer)
    return window_based_shuffle(out_data, window_size=window_size, seed=seed)
