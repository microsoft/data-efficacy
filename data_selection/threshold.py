def select(in_data, args):
    score_field = args.score_field
    threshold = args.threshold

    out_data = list()
    for x in in_data:
        if x[score_field] > threshold:
            out_data.append(x) 

    return out_data
