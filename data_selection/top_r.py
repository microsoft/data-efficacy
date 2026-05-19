def select(in_data, args):
    score_field = args.score_field
    r = float(max(0, args.r))
    k = int(len(in_data) * r)

    try:
        out_data = sorted(in_data, key=lambda x: x[score_field], reverse=False)
        if k < len(out_data):
            out_data = out_data[-k:]
    except TypeError as e:
        raise ValueError(f"Error sorting data by field '{score_field}': {e}")

    return out_data
