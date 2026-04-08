import random
import numpy as np
import warnings

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
    Segment Ordering：按分数重排序数据，将数据分为前、中、后三段，并分别打乱

    Args:
        in_data (list): 输入数据列表，每个元素为带有分数的字典
        args: 包含配置参数的对象
            - score_field: 分数字段名
            - x_pct: 前段百分比 (0-100)
            - y_pct: 后段百分比 (0-100)
            - front_is_high: 前段是否取高分样本
            - back_is_high: 后段是否取高分样本
            - seed: 随机种子（可选）

    Returns:
        list: 重排序后的数据列表
    """
    score_field = args.score_field
    total_samples = len(in_data)

    if hasattr(args, 'seed'):
        random.seed(args.seed)
        np.random.seed(args.seed)

    sorted_data = sorted(
        enumerate(in_data),
        key=lambda x: (x[1][score_field], x[0]),
        reverse=False
    )
    sorted_data = [item[1] for item in sorted_data]

    n_front = int(np.floor(args.x_pct / 100 * total_samples))
    n_back = int(np.floor(args.y_pct / 100 * total_samples))

    total_selected = n_front + n_back
    if total_selected > total_samples:

        ratio = total_samples / total_selected
        n_front = int(np.floor(n_front * ratio))
        n_back = int(np.floor(n_back * ratio))
        warnings.warn(
            f"前段({args.x_pct}%)和后段({args.y_pct}%)的总和超过100%! "
            f"已按比例缩减为前段{n_front}个、后段{n_back}个样本。"
        )
        total_selected = n_front + n_back

    front = []
    back = []
    middle = []

    if args.front_is_high == args.back_is_high:

        if args.front_is_high:
            selected = sorted_data[-total_selected:] if total_selected > 0 else []
            middle = sorted_data[:-total_selected] if total_selected > 0 else sorted_data
        else:
            selected = sorted_data[:total_selected] if total_selected > 0 else []
            middle = sorted_data[total_selected:] if total_selected > 0 else sorted_data

        if selected:
            random.shuffle(selected)

            front = selected[:n_front]
            back = selected[n_front:total_selected]
    else:

        if args.front_is_high:
            front = sorted_data[-n_front:] if n_front > 0 else []
            remaining = sorted_data[:-n_front] if n_front > 0 else sorted_data
        else:
            front = sorted_data[:n_front] if n_front > 0 else []
            remaining = sorted_data[n_front:] if n_front > 0 else sorted_data

        if args.back_is_high:
            back = remaining[-n_back:] if n_back > 0 else []
            middle = remaining[:-n_back] if n_back > 0 else remaining
        else:
            back = remaining[:n_back] if n_back > 0 else []
            middle = remaining[n_back:] if n_back > 0 else remaining


    random.shuffle(front)
    random.shuffle(middle)
    random.shuffle(back)
    out_data = front + middle + back

    return out_data