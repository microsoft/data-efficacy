import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import deepspeed
from trainer import train
from utils import load_yaml, init, add_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training.")
    parser.add_argument("--local_rank", type=int, help="Local rank for deepspeed.", default=0)
    parser.add_argument("--data_path", type=str, required=True, help="The path of training data.")
    parser.add_argument("--model_path", type=str, required=True, help="The input path of model.")
    parser.add_argument("--save", type=str, required=True, help="The save path of model.")
    parser.add_argument("--method", type=str, choices=["pretrain", "posttrain"], default="pretrain", help="Training type: 'pretrain' and 'posttrain'.")
    parser.add_argument("--config_path", type=str, default="./model_train/config/train.yaml", help="Config file for additional parameters (YAML format).")

    args = parser.parse_args()
    parser = deepspeed.add_config_arguments(parser)
    args = add_args(args, load_yaml(args.config_path))

    init(args)
    if args.method in ["pretrain", "posttrain"]:
        train(args)
