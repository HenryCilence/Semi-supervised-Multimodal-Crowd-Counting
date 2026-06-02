import os
import random

def select_labeled_samples_fixed(image_dir, num_samples):
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('_RGB.jpg')])
    total_files = len(all_files)
    
    if total_files <= num_samples:
        return all_files
    
    indices = [i * total_files // num_samples for i in range(num_samples)]
    selected_files = [all_files[i] for i in indices]
    
    return selected_files

image_dir = r""
save_dir = r"./drone-100.txt"
num_labeled_samples = 9999  
labeled_files = select_labeled_samples_fixed(image_dir, num_labeled_samples)

with open(save_dir, "w") as f:
    for file in labeled_files:
        f.write(file + "\n")

print(f"Selected {len(labeled_files)} labeled samples.")