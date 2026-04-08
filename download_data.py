import os
import json
from datasets import load_dataset

# 1. 下载并加载完整数据集
# 首次运行会完整下载数据并缓存在本地（~/.cache/huggingface/datasets），需要一些时间
print("正在下载并加载完整数据集（首次运行需要耐心等待）...")
ds = load_dataset("nvidia/OpenCodeInstruct", split="train")

# 2. 全局随机打乱
# 因为数据已经全部在本地，这里不再需要 buffer_size，直接全局 shuffle
print("正在进行全局随机打乱...")
shuffled_ds = ds.shuffle(seed=42)

# 3. 截取前 100,000 条
# 使用 select() 方法根据索引快速切片
print("正在截取前 100,000 条数据...")
sampled_data = shuffled_ds.select(range(100000))

# 4. 确保输出目录存在
output_path = "data/opencode_10w.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 5. 保存到本地 JSONL 文件
print(f"正在写入数据到 {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    for i, example in enumerate(sampled_data):
        f.write(json.dumps(example, ensure_ascii=False) + "\n")
        if (i + 1) % 10000 == 0:
            print(f"已完成: {i + 1}/100000")

print("🎉 采样完成！这次的数据是 100% 全局随机的。")
# from datasets import load_dataset
# import json
# import os

# # 1. 确保目录存在
# os.makedirs("data", exist_ok=True)

# # 2. 流式加载 FineWeb-Edu [cite: 2026-03-08]
# # 注意：FineWeb-Edu 非常巨大，必须使用 streaming=True
# print("正在连接 Hugging Face...")
# ds = load_dataset("HuggingFaceFW/fineweb-edu", name="default", split="train", streaming=True)

# # 3. 随机采样 100,000 条
# shuffled_ds = ds.shuffle(seed=42, buffer_size=10000)
# sampled_data = shuffled_ds.take(100000)

# output_path = "data/fineweb.jsonl"
# print(f"正在写入数据到 {output_path}...")

# with open(output_path, "w", encoding="utf-8") as f:
#     for i, example in enumerate(sampled_data):
#         try:
#             f.write(json.dumps(example, ensure_ascii=False) + "\n")
#             if (i + 1) % 10000 == 0:
#                 print(f"已完成: {i + 1}/100000")
#         except Exception as e:
#             print(f"处理第 {i} 条数据时出错: {e}")
#             continue

# print(f"采样完成！文件已保存至 {output_path}")