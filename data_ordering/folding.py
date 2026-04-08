import numpy as np

def window_based_shuffle(data, window_size, seed=42):
    """
    Jittering Ordering:对列表进行局部窗口内的随机打乱,整体有序，局部无序

    Args:
        data:输入数据列表
        window_size:局部打乱窗口大小,如果为 0 或 1，则不进行局部打乱
        seed:随机种子

    Returns:
        list: 重排序后的数据列表
    """
    if window_size <= 1:
        return data

    n = len(data)
    rng = np.random.RandomState(seed)
    shuffled_final_data = []

    for i in range(0, n, window_size):
        chunk = data[i: i + window_size]
        rng.shuffle(chunk)
        shuffled_final_data.extend(chunk)

    return shuffled_final_data


def order(in_data, args):
    """
    Folding Ordering：将输入数据按分数进行升序排列，排序后的序列按 folding_layer 进行取模分桶，依次提取每个桶中的元素并拼接，实现分数的跳跃式分布。
    最后可选执行局部窗口打乱。

    Args:
        in_data (list): 输入数据列表，每个元素为带有分数的字典
        args: 包含配置参数的对象
            - score_field: 分数字段名
            - folding_layer：折叠层数
            - window_size: 局部打乱窗口大小 (可选)
            - seed: 随机种子 (可选)

    Returns:
        list: 重排序后的数据列表
    """
    score_field = args.score_field
    layers = args.folding_layer

    window_size = getattr(args, "window_size", 0)
    seed = getattr(args, "seed", 42)
    ascending = getattr(args, "ascending", True)

    sorted_data = sorted(in_data, key=lambda x: x[score_field], reverse=not ascending)

    out_data = list()
    for l in range(layers):
        sub_data = [sorted_data[i] for i in range(len(sorted_data)) if i % layers == l]
        out_data.extend(sub_data)

    if window_size > 1:
        out_data = window_based_shuffle(out_data, window_size, seed)

    return out_data
