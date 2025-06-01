import sys, pandas as pd, pathlib, os

csv_file = pathlib.Path("任务一标签（新）.csv")  # 输出标签目录
lbl_dir = pathlib.Path("meter/labels")  # 输出标签目录
H = 296
W = 400
df = pd.read_csv(csv_file)



os.makedirs(lbl_dir, exist_ok=True)
for _, row in df.iterrows():
    # 归一化
    cx = (row.xmin + row.xmax) / 2 / W
    cy = (row.ymin + row.ymax) / 2 / H
    w  = (row.xmax - row.xmin) / W
    h  = (row.ymax - row.ymin) / H

    txt = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
    (pathlib.Path(lbl_dir) / (pathlib.Path(row.filename).stem + ".txt")).write_text(txt)

