import random
import warnings

try:
    from .common import sort_data
except ImportError:
    from common import sort_data


def _take_boundary(sorted_data, n_items, high):
    if n_items <= 0:
        return [], list(sorted_data)
    if high:
        return list(sorted_data[-n_items:]), list(sorted_data[:-n_items])
    return list(sorted_data[:n_items]), list(sorted_data[n_items:])


def order(in_data, args):
    """Segment Ordering (SEG).

    The front and back segments draw from low-score or high-score boundaries,
    while the remaining samples form the middle segment. Each segment is
    shuffled independently.
    """
    score_field = args.score_field
    total_samples = len(in_data)
    seed = getattr(args, "seed", 42)
    rng = random.Random(seed)

    sorted_data = sort_data(in_data, score_field, ascending=True)
    n_front = int(total_samples * args.x_pct // 100)
    n_back = int(total_samples * args.y_pct // 100)
    total_selected = n_front + n_back

    if total_selected > total_samples:
        ratio = total_samples / total_selected
        n_front = int(n_front * ratio)
        n_back = total_samples - n_front
        warnings.warn(
            f"x_pct + y_pct exceeds 100; resized to {n_front} front and {n_back} back samples.",
            RuntimeWarning,
        )

    front_is_high = bool(args.front_is_high)
    back_is_high = bool(args.back_is_high)

    if front_is_high == back_is_high:
        selected, middle = _take_boundary(sorted_data, n_front + n_back, high=front_is_high)
        rng.shuffle(selected)
        front = selected[:n_front]
        back = selected[n_front:]
    else:
        front, remaining = _take_boundary(sorted_data, n_front, high=front_is_high)
        back, middle = _take_boundary(remaining, n_back, high=back_is_high)

    rng.shuffle(front)
    rng.shuffle(middle)
    rng.shuffle(back)
    return front + middle + back
