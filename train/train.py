from ultralytics import YOLO
import yaml
import argparse
import os



def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser(description="yolo_Dect")
    args.add_argument("--yaml_name",default="yoloDe.yaml")
    return args.parse_args()

def main():
    args = parse_args()
    configs = os.path.join("../config",args.yaml_name)
    with open(configs, "r") as f:
        cfg = yaml.safe_load(f)

    # 1) 加载预训练模型
    model = YOLO(f"{cfg['base_path']}/{cfg['model_path']}")
    # 2) 开始 finetune
    model.train(
        data=f"{cfg['base_path']}/{cfg['data_path']}",
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=cfg['batch'],
        device=cfg['device'],
        freeze=cfg['freeze']
    )
    # 3) 评估 test 集
    metrics = model.val(split="test")   # or split='val'
    print(metrics.box.map50)            # 举例：输出 mAP50

    # 5. 预测一张新图（可选）
    # result = model(f"{cfg['base_path']}/meter/images/test/your_img.jpg", conf=0.25)
    # result[0].show()                  # 可视化




if __name__ == "__main__":
    main()