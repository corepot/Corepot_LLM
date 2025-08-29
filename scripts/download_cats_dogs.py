import os
import requests
from pathlib import Path

# 数据目录
BASE_DIR = Path("/root/corepot/data")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# 创建目录
for d in [TRAIN_DIR / "cat", TRAIN_DIR / "dog", VAL_DIR / "cat", VAL_DIR / "dog"]:
    d.mkdir(parents=True, exist_ok=True)

# 示例猫狗图片 URL 列表（可自行增加）
cat_urls = [
    "https://cdn.pixabay.com/photo/2017/11/09/21/41/cat-2934720_1280.jpg",
    "https://cdn.pixabay.com/photo/2018/07/13/11/18/cat-3535405_1280.jpg",
    "https://cdn.pixabay.com/photo/2019/02/03/19/06/cat-3978285_1280.jpg"
]

dog_urls = [
    "https://cdn.pixabay.com/photo/2015/03/26/09/54/dog-690176_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/09/25/13/12/dog-2785074_1280.jpg",
    "https://cdn.pixabay.com/photo/2018/03/06/22/47/dog-3201324_1280.jpg"
]

# 下载函数
def download_images(urls, save_dir, prefix):
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                ext = ".jpg"
                path = save_dir / f"{prefix}_{i}{ext}"
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"下载成功: {path}")
            else:
                print(f"下载失败: {url}, 状态码: {r.status_code}")
        except Exception as e:
            print(f"下载异常: {url}, 错误: {e}")

# 下载训练集
download_images(cat_urls, TRAIN_DIR / "cat", "cat")
download_images(dog_urls, TRAIN_DIR / "dog", "dog")

# 下载验证集
download_images(cat_urls, VAL_DIR / "cat", "cat")
download_images(dog_urls, VAL_DIR / "dog", "dog")

print("下载完成，训练和验证集已准备好！")
