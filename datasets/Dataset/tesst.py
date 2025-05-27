import pandas as pd
import cv2
from pathlib import Path

# =======================
# 0. 配置路径
# =======================
CSV_PATH   = 'labels.csv'          # 你的 CSV 文件
OUT_FOLDER = Path('boxed')         # 输出描框图像目录
OUT_FOLDER.mkdir(exist_ok=True)

# =======================
# 1. 读取 CSV
# =======================
df = pd.read_csv(CSV_PATH,
                 names=['filename', 'number', 'xmin', 'ymin', 'xmax', 'ymax'])

# =======================
# 2. 遍历逐行画框
# =======================
row = df.iloc[2,:]
img_path = row.filename

# 2.1 读图（BGR）
img = cv2.imread(str(img_path))
print(img.shape)
# 2.2 画矩形框
pt1, pt2 = (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax))
cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=2)

# 2.3 可选：在左上角写读数
cv2.putText(img, str(row.number), (pt1[0], pt1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 2.4 保存到输出目录
cv2.imwrite(str(OUT_FOLDER / row.filename), img)
