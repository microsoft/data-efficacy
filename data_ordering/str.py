import numpy as np
def _apply_interleave_fold(data_segment, score_field, layers, reverse_even_layers=False): # <--- MODIFICATION: 增加新参数
    """
    应用 "Folding" 逻辑
    
    Args:
        data_segment (list): 要处理的数据片段
        score_field (str): 分数字段
        layers (int): 交叉的层数
        reverse_even_layers (bool): 是否翻转偶数层（实现 Zigzag）
    """
    if not data_segment:
        return []
        

    sorted_data = sorted(data_segment, key=lambda x: x[score_field], reverse=False)
    
    out_data = list()
    for l in range(layers):

        sub_data = [sorted_data[i] for i in range(len(sorted_data)) if i % layers == l]
        if reverse_even_layers and (l % 2 != 0):
            sub_data.reverse()
        out_data.extend(sub_data)
        
    return out_data


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
    Stair Ordering：先全局排序，然后在 K-1 个分割点应用局部折叠，最后局部窗口打乱（可选）

    Args:
        in_data (list): 输入数据列表，每个元素为带有分数的字典。
        args: 包含配置参数的对象。
            - score_field: 分数字段名
            - ascending: 是否升序
            - reverse_even_layers:是否翻转偶数层（参数默认False）
            - folding_layer: 局部折叠参数 (来自 'folding')
            - num_section: 数据被分成的总折数（参数默认为2）
            - folding_ratio: 在分割点处，向上和向下各取多少比例的数据进行折叠 (例如 0.10 表示各 10%)
            - window_size:局部打乱窗口大小,如果为 0 或 1，则不进行局部打乱
            - seed:随机种子

    Returns:
        list: 重排序后的数据列表
    """


    score_field = args.score_field
    ascending = args.ascending
    num_sections = args.num_sections
    folding_ratio = args.folding_ratio
    interleave_layers = args.folding_layer
    reverse_even_layers = getattr(args, 'reverse_even_layers', False)

    window_size = getattr(args, "window_size", 0)
    seed = getattr(args, "seed", 42)

    if ascending:
        sorted_data = sorted(in_data, key=lambda x: x[score_field], reverse=False)
    else:
        sorted_data = sorted(in_data, key=lambda x: x[score_field], reverse=True)
    N = len(sorted_data)

    if N == 0:
        return sorted_data

    if num_sections > 1:
        split_indices = [int(round(N * i / num_sections)) for i in range(1, num_sections)]
        radius_items = int(round(N * folding_ratio))
        segments = []
        current_index = 0

        for sp_index in split_indices:
            fold_start = max(0, sp_index - radius_items)
            fold_end = min(N, sp_index + radius_items)
            fold_start = max(fold_start, current_index)
            fold_end = max(fold_end, fold_start)
            if fold_start > current_index:
                segments.append((current_index, fold_start, 'stable'))
            if fold_end > fold_start:
                segments.append((fold_start, fold_end, 'fold'))
            current_index = fold_end

        if current_index < N:
            segments.append((current_index, N, 'stable'))

        out_data = list()
        for start, end, segment_type in segments:
            data_segment = sorted_data[start:end]

            if not data_segment:
                continue

            if segment_type == 'stable':
                out_data.extend(data_segment)
            else:
                folded_segment = _apply_interleave_fold(
                    data_segment,
                    score_field,
                    interleave_layers,
                    reverse_even_layers
                )
                out_data.extend(folded_segment)
    else:
        out_data = sorted_data

    if window_size > 1:
        out_data = window_based_shuffle(out_data, window_size, seed)

    return out_data