import os, random, shutil
from pathlib import Path

IMG_DIR = Path("Dataset")     # 原图像目录（你图都在这）
LBL_DIR = Path("meter/labels")         # YOLO 标签已在这
SAVE_ROOT = Path("meter")              # 输出根目录

for split in ["train", "val", "test"]:
    os.makedirs(SAVE_ROOT / f"labels/{split}", exist_ok=True)

# 拆分比例
train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1


# 获取图像文件列表（只管 .jpg）
image_files = sorted(list(Path(os.path.join(IMG_DIR, f)) for f in os.listdir(IMG_DIR) if f.endswith(".jpg")))
random.shuffle(image_files)

n = len(image_files)
n_train = int(n * train_ratio)
n_val   = int(n * val_ratio)

splits = {
    "train": image_files[:n_train],
    "val":   image_files[n_train:n_train+n_val],
    "test":  image_files[n_train+n_val:]
}

# 复制图片和对应的标签文件
for split, files in splits.items():
    for img_path in files:
        lbl_path = LBL_DIR / img_path.with_suffix(".txt").name
        shutil.copy2(img_path, SAVE_ROOT / f"images/{split}" / img_path.name)
        shutil.copy2(lbl_path, SAVE_ROOT / f"labels/{split}" / lbl_path.name)

print("✅ 拆分完毕！train/val/test 数据已生成")
