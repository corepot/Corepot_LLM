import os
import shutil
from pathlib import Path

def prepare_dataset(source_dir, target_dir, label_file):
    """
    根据标签文件，把图片从 source_dir 移动或复制到 target_dir 的对应分类文件夹。
    
    参数：
    - source_dir: 原始图片存放路径（图片和标签文件）
    - target_dir: 整理后分类存放的根目录，自动创建子文件夹
    - label_file: 标签文件，格式为每行 "图片文件名 类别名"，例如：
      img001.jpg cat
      img002.png dog
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        filename, label = line.split()
        src_path = os.path.join(source_dir, filename)
        label_dir = os.path.join(target_dir, label)
        dst_path = os.path.join(label_dir, filename)

        if not os.path.exists(src_path):
            print(f"警告：源文件不存在 {src_path}")
            continue

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        shutil.copy2(src_path, dst_path)  # 复制文件，如果你想移动改用 shutil.move

    print("数据集整理完成。")

if __name__ == "__main__":
    # 举例用法，请根据实际修改路径
    source_dir = "raw_images"
    target_train_dir = "data/train"
    target_val_dir = "data/val"
    train_label_file = "train_labels.txt"
    val_label_file = "val_labels.txt"

    prepare_dataset(source_dir, target_train_dir, train_label_file)
    prepare_dataset(source_dir, target_val_dir, val_label_file)







