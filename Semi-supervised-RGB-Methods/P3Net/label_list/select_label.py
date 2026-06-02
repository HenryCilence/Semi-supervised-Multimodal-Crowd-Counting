import os
import random

# 精确选择前N个等间隔样本（不使用随机起始点）
def select_labeled_samples_fixed(image_dir, num_samples):
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('_RGB.jpg')])
    total_files = len(all_files)
    
    if total_files <= num_samples:
        return all_files
    
    # 精确计算间隔
    indices = [i * total_files // num_samples for i in range(num_samples)]
    selected_files = [all_files[i] for i in indices]
    
    return selected_files

# 使用示例
image_dir = r"/media/dataset/person_dataset/multi-modal_crowd_counting/Drone/train"
save_dir = r"/home/home/menghaoliang/code/count3/Multi-semi/label_list/drone-40.txt"
num_labeled_samples = 722  # 需要选择的有标签样本数量
labeled_files = select_labeled_samples_fixed(image_dir, num_labeled_samples)

# 保存结果到文件（只需要文件名，不需要路径）
with open(save_dir, "w") as f:
    for file in labeled_files:
        f.write(file + "\n")  # 只写入文件名，如 "0001_RGB.jpg"

print(f"已选择 {len(labeled_files)} 个有标签样本")