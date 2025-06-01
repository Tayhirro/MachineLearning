import os
import argparse
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
import cv2

def infer_image(model, image_path):
    """
    使用 YOLO 模型对单张图像进行推理。
    """
    return model(image_path)
def crop_and_save_images(results, dest_dir):
    """
    根据检测结果裁剪图像并保存。
    """
    # 遍历每个检测到的目标
    for i, result in enumerate(results):
        # 获取图像路径
        image_path = result.path

        # 读取原始图像
        src_img = cv2.imread(image_path)
        
        if src_img is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        # 获取检测结果中的边界框
        boxes = result.boxes
        
        # 遍历每个检测到的目标
        for j, box in enumerate(boxes):
            # 提取边界框坐标 [x1, y1, x2, y2]
            print(box)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # 裁剪图像
            cropped_image = src_img[y1:y2, x1:x2]
            # 构建保存路径 
            filename = os.path.basename(image_path)
            save_path = os.path.join(dest_dir, f"{os.path.splitext(filename)[0]}_cropped_{j}.jpg")
            
            # 保存裁剪后的图像
            cv2.imwrite(save_path, cropped_image)
            print(f"Saved cropped image to {save_path}")

def copy_and_handle_images(src_dir, dest_dir):
    """
    处理文件中所有图，裁剪为目标区域
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for filename in os.listdir(src_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                # 使用 YOLO 模型进行推理
                results = infer_image(model, src_path)
                # 裁剪并保存图像
                crop_and_save_images(results,dest_dir)
            except Exception as e:
                print(f"Error processing {src_path}: {e}")




model = YOLO("../best.pt")  # Load a pretrained YOLO model

data_dir_base = "../../datasets/meter/images/"

for split in ["train", "val", "test"]:
    data_dir = os.path.join(data_dir_base, split)
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        continue
    dest_base_dir = "../../datasets/classify/"
    if os.path.exists(os.path.join(dest_base_dir, split)):
        print(f"Directory {dest_base_dir} already exists. Skipping copy for {split}.")
    else:
        os.makedirs(dest_base_dir, exist_ok=True)
        print(f"Copying images from {data_dir} to {dest_base_dir}")
        copy_and_handle_images(data_dir, os.path.join(dest_base_dir, split))