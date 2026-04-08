import math
import numpy as np

def gumbel_indices_sort(sorted_indices, tau=1.0, use_gumbel=False, seed=42):
    """
    对已经按真实分数排序后的索引序列进行“排名位置”的 Gumbel 扰动，再据此重排索引
    这消除了分数分布对随机性的影响，随机性仅由 tau 决定

    Args:
        sorted_indices (Sequence[int]): 已由真实分数升序排序得到的索引列表
        tau (float): 温度系数，控制扰动强度。越大越随机
        use_gumbel (bool): 是否启用 Gumbel 扰动。False 时保持输入顺序不变
        seed (int): 随机种子，保证可复现

    Returns:
        List[int]: 经过 Gumbel 扰动后的新索引序列
    """
    idx = np.array(sorted_indices)
    n = idx.shape[0]
    np.random.seed(seed)
    if use_gumbel:
        gumbel_noise = -np.log(-np.log(np.random.rand(n)))
        perturbed_positions = np.arange(n, dtype=float) + gumbel_noise * tau
    else:
        perturbed_positions = np.arange(n, dtype=float)

    new_order = np.argsort(perturbed_positions)
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
    Zigzag Ordering:先按真实分数进行确定性排序得到 sorted_indices，再用 gumbel_indices_sort
    对 indices 的“排名位置”加 Gumbel 噪声(可选)，局部窗口打乱（可选），最后执行Zigzag折叠逻辑

     Args:
        in_data (list): 输入数据列表，每个元素为带有分数的字典
        args: 包含配置参数的对象
            - score_field: 分数字段名
            - zigzag_layer: zigzag折叠层数
            - use_gumble: 是否启用gumble扰动
            - temperature: 温度系数，控制扰动强度。越大越随机
            - ascending: 是否升序（True为升序，False为降序）
            - seed: 随机种子（可选）
            - window_size:局部打乱窗口大小,如果为 0 或 1，则不进行局部打乱

    Returns:
        list[int]: 经过 Gumbel 扰动后的新索引序列
    """
    score_field = args.score_field
    zigzag_layer = args.zigzag_layer
    tau = args.temperature
    use_gumbel = args.use_gumbel
    seed = args.seed
    window_size = getattr(args, "window_size", 0)
    ascending = getattr(args, "ascending", True)

    scores = np.array([item[score_field] for item in in_data])
    base_sorted_indices = list(np.argsort(scores))
    if not ascending:
        base_sorted_indices = base_sorted_indices[::-1]
    sorted_indices = gumbel_indices_sort(base_sorted_indices, tau, use_gumbel, seed)

    r = 2
    out_data = []
    n = len(sorted_indices)
    for l in range(zigzag_layer):
        sub_data = [in_data[sorted_indices[i]] for i in range(n) if i % zigzag_layer == l]
        sub_data_drop = [sub_data[i] for i in range(len(sub_data)) if i % r == 0][::-1]
        sub_data_rise = [sub_data[i] for i in range(len(sub_data)) if i % r != 0]
        out_data.extend(sub_data_drop)
        out_data.extend(sub_data_rise)

    if window_size > 1:
        out_data = window_based_shuffle(out_data, window_size, seed)

    return out_data

