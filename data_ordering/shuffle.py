import random

def order(in_data, args):
    """
      随机排序

      Args:
          in_data (list): 输入数据列表，每个元素为带有分数的字典
          args: 包含配置参数的对象
              - score_field: 分数字段名
              - seed：随机种子（可选）

      Returns:
          list: 重排序后的数据列表
      """
    random.seed(args.seed)
    out_data = random.sample(in_data, len(in_data))
    return out_data
