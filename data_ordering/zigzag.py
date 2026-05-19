try:
    from .common import sort_data, window_based_shuffle, zigzag_order
except ImportError:
    from common import sort_data, window_based_shuffle, zigzag_order


def order(in_data, args):
    score_field = args.score_field
    zigzag_layer = args.zigzag_layer
    ascending = getattr(args, "ascending", True)
    window_size = getattr(args, "window_size", 0)
    seed = getattr(args, "seed", 42)

    sorted_data = sort_data(in_data, score_field, ascending=ascending)
    out_data = zigzag_order(sorted_data, zigzag_layer)
    return window_based_shuffle(out_data, window_size=window_size, seed=seed)
