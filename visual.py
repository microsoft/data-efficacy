import json
import heapq
import matplotlib.pyplot as plt
import numpy as np
DATA_NAME = "opencode_antisorting"
INPUT_FILE = f"./data/{DATA_NAME}.jsonl"
OUTPUT_IMAGE = f"./photo/{DATA_NAME}.png"
SCORE =f"average_test_score"


def visualize_score(jsonl_file):
    """
    读取 JSONL 文件中的 "score" 字段，并按文件顺序进行可视化。

    Args:
        jsonl_file (str): JSONL 文件路径。
    """
    # 读取 JSONL 文件
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]

    # 检查字段是否存在
    if not all(SCORE in item for item in data):
        raise KeyError("字段 socre 不存在于某些样本中！")

    # 提取 "score" 字段值
    scores = [item[SCORE] for item in data]

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores)), scores, marker='o', linestyle='-', color='b', alpha=0.7)

    # 添加标题和标签
    plt.title("Score Visualization", fontsize=16)
    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Score", fontsize=14)

    # 显示网格
    plt.grid(axis='both', linestyle='--', alpha=0.7)

    # 显示图表
    plt.tight_layout()
    # plt.show()
    # plt.savefig('./vis/finewebedu_score_visualization.png')
    plt.savefig(OUTPUT_IMAGE)


if __name__ == "__main__":
    visualize_score(INPUT_FILE)