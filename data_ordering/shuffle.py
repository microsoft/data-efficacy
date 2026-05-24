import random


def order(in_data, args):
    seed = getattr(args, "seed", 42)
    rng = random.Random(seed)
    out_data = list(in_data)
    rng.shuffle(out_data)
    return out_data
