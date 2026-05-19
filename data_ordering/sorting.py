try:
    from .common import gumbel_rank_jitter, sorted_indices, window_based_shuffle
except ImportError:
    from common import gumbel_rank_jitter, sorted_indices, window_based_shuffle


def order(in_data, args):
    score_field = args.score_field
    ascending = getattr(args, "ascending", True)
    tau = getattr(args, "temperature", 1.0)
    use_gumbel = getattr(args, "use_gumbel", False)
    seed = getattr(args, "seed", 42)
    window_size = getattr(args, "window_size", 0)

    indices = sorted_indices(in_data, score_field, ascending=ascending)
    indices = gumbel_rank_jitter(indices, tau=tau, use_gumbel=use_gumbel, seed=seed)
    sorted_data = [in_data[idx] for idx in indices]
    return window_based_shuffle(sorted_data, window_size=window_size, seed=seed)
