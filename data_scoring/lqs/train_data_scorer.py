import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import torch
import deepspeed

from utils import load_yaml, init, init_deepspeed, add_args
from data_scoring.lqs.tools.get_name_lqs import s4_1_annotated_proxy_token_data, s5_scorer_model, s3_proxy_token_data, s1_2_full_token_data, s2_2_target_token_data, s4_3_prepare_scorer_data, s4_2_prepare_scorer_model, s1_1_full_target_token_model
from data_scoring.lqs.annotation.trainer import GammaTrainer
from data_scoring.lqs.scorer.trainer import DataScorerTrainer

torch.set_num_threads(16)


def train(args):
    torch.backends.cudnn.enabled = False
    
    init(args)
    ds_config = init_deepspeed(args)

    os.makedirs(args.save, exist_ok=True)
    
    if args.type == "annotation_data":
        trainer = GammaTrainer(args)
    elif args.type == "scoring_data":
        trainer = DataScorerTrainer(args, ds_config, args.do_train)
    else:
        raise ValueError(f"Invalid type: {args.type}")    
    
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of LQS data scorer.") 
    parser.add_argument("--local_rank", type=int, help="Local rank for deepspeed.", default=0)
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data_training", "scorer_data_infer"], default="scorer_data_training", help="The running step of LQS.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    process_param = load_yaml(args.config_path)
    args = add_args(args, process_param, args.lqs_process)

    if args.lqs_process == "annotation_data":
        args.data_dir = os.path.join(args.data_dir, s1_2_full_token_data(process_param))
        args.model_path = os.path.join(args.model_path, s1_1_full_target_token_model(process_param))
        args.dev_data_dir = os.path.join(args.dev_data_dir, s2_2_target_token_data(process_param), "dev")
        args.proxy_data_dir = os.path.join(args.proxy_data_dir, s3_proxy_token_data(process_param))
        args.save = os.path.join(args.save, s4_1_annotated_proxy_token_data(process_param))
    elif args.lqs_process == "scorer_data_training":
        args.data_dir = os.path.join(args.data_dir, s4_3_prepare_scorer_data(process_param))
        args.model_path = os.path.join(args.model_path, s4_2_prepare_scorer_model(process_param))
        args.save = os.path.join(args.save, s5_scorer_model(process_param))
    else:
        raise ValueError(f"Error lqs process for this file: {args.lqs_process}")    
    
    train(args)
