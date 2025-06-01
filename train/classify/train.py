import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import yaml



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classification model")
    parser.add_argument("--yaml_name", default="classify.yaml", help="Configuration file name")
    return parser.parse_args()


def main(): 
    args = parse_args()
    

if __name__ == "__main__":
    main()

