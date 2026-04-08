import numpy as np


def gumbel_indices_sort(sorted_indices, tau=1.0, use_gumbel=False, seed=42, ascending=True):
    """
    对已经按真实分数排序后的索引序列进行“排名位置”的 Gumbel 扰动，再据此重排索引。

    Args:
        sorted_indices (list[int]): 已由真实分数排序得到的索引列表
        tau (float): 温度系数（控制扰动强度，越大越随机）
        use_gumbel (bool): 是否启用 Gumbel 扰动
        seed (int): 随机种子
        ascending (bool): 排序方向，True=升序，False=降序

    Returns:
        list[int]: 加入 Gumbel 扰动后的新索引序列
    """
    idx = np.array(sorted_indices)
    n = idx.shape[0]

    np.random.seed(seed)
    if use_gumbel:
        gumbel_noise = -np.log(-np.log(np.random.rand(n)))
        perturbed_positions = np.arange(n, dtype=float) + gumbel_noise * tau
    else:
        perturbed_positions = np.arange(n, dtype=float)

    if ascending:
        new_order = np.argsort(perturbed_positions)
    else:
        new_order = np.argsort(-perturbed_positions)

    return list(idx[new_order])


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
    Sorting Ordering：先按真实分数确定性进行sorting排序 再在索引顺序上加入 Gumbel 噪声（可选），局部窗口打乱（可选）

    Args:
        in_data (list): 输入数据列表，每个元素为带有分数的字典。
        args: 包含配置参数的对象。
            - score_field: 分数字段名
            - ascending: 决定是否升序
            - use_gumbel (bool): 是否启用 Gumbel 扰动。False 时保持输入顺序不变
            - temperature: 温度系数（控制扰动强度，越大越随机）
            - window_size:局部打乱窗口大小,如果为 0 或 1，则不进行局部打乱
            - seed (int): 随机种子

    Returns:
        list: 重排序后的数据列表
    """
    score_field = args.score_field
    ascending = args.ascending


    tau = getattr(args, "temperature", 1.0)
    use_gumbel = getattr(args, "use_gumbel", False)
    seed = getattr(args, "seed", 42)
    window_size = getattr(args, "window_size", 0)


    scores = np.array([item[score_field] for item in in_data])
    base_sorted_indices = list(np.argsort(scores))
    if not ascending:
        base_sorted_indices = base_sorted_indices[::-1]
    sorted_indices = gumbel_indices_sort(base_sorted_indices, tau, use_gumbel, seed, ascending=True)
    sorted_data = [in_data[i] for i in sorted_indices]


    if window_size > 1:
        final_data = window_based_shuffle(sorted_data, window_size, seed)
    else:
        final_data = sorted_data

    return final_data
