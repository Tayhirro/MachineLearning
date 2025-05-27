from ultralytics import YOLO

# 1) 加载预训练模型
model = YOLO("/home/jiahongyu/Detection/train/yolov8n.pt")

# 2) 开始 finetune
model.train(
    data="/home/jiahongyu/Detection/datasets/meter/meter.yaml",  # 数据集配置文件
    epochs=100,
    imgsz=640,
    batch=16,
    device="0,1,2,3",  # 使用多卡训练
    freeze=10           # 冻结前10层
)

# 3) 评估 test 集
metrics = model.val(split="test")   # or split='val'
print(metrics.box.map50)            # 举例：输出 mAP50

# # 4) 预测一张新图
# result = model("meter/images/test/your_img.jpg", conf=0.25)
# result[0].show()                    # 可视化
