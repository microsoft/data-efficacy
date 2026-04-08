import json
import random
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ================= 配置区（类似于宏） =================
# 你每次只需要改这里：
DATA_NAME = "opencode_foldingl3_50000" 
INPUT_FILE = f"./data/{DATA_NAME}.jsonl"
OUTPUT_IMAGE = f"./photo/{DATA_NAME}.png"
SCORE = f"average_test_score"
SAMPLE_SIZE = 500
# =====================================================

def visualize_score(jsonl_file, n):
    # 自动创建保存图片的文件夹，防止报错
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    
    reservoir = []  # 存元组: (original_index, score)
    
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                item = json.loads(line)
                if SCORE not in item: continue
                
                current_score = float(item[SCORE])
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
                
            if len(reservoir) < n:
                reservoir.append((i, current_score))
            else:
                r = random.randint(0, i)
                if r < n:
                    reservoir[r] = (i, current_score)

    reservoir.sort(key=lambda x: x[0]) 
    scores = [x[1] for x in reservoir]

    plt.figure(figsize=(6, 6))
    
    # 颜色映射：0(红) -> 0.5(金) -> 1(绿)
    colors = ["orangered", "gold", "green"]
    custom_cmap = LinearSegmentedColormap.from_list("orange_green", colors)

    # 绘制散点图
    plt.scatter(range(len(scores)), scores, c=scores, cmap=custom_cmap, alpha=0.8)

    plt.xlabel("Index", fontsize=20)
    plt.ylabel(SCORE, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"图像已成功保存至: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    # 使用宏定义的路径和采样数
    visualize_score(INPUT_FILE, SAMPLE_SIZE)
