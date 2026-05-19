import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
from utils import load_yaml, add_args
from data_scoring.lqs.tools.get_name_lqs import s1_2_full_token_data, s5_scorer_model
from data_scoring.lqs.scorer.infer import data_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of LQS data scorer.") 
    parser.add_argument("--local_rank", type=int, help="Local rank for deepspeed.", default=0)
    parser.add_argument("--output-data-path", type=str, required=True, help="Output jsonl data path.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data_training", "scorer_data_infer"], default="scorer_data_infer", help="The running step of LQS.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    process_param = load_yaml(args.config_path)
    args = add_args(args, process_param, args.lqs_process)

    if args.lqs_process == "scorer_data_infer":
        args.model_path = os.path.join(args.model_path, s5_scorer_model(process_param))
        args.input_data_path = os.path.join(args.bin_data_path, s1_2_full_token_data(process_param)) + '.jsonl'
    else:
        raise ValueError(f"Error lqs process for this file: {args.lqs_process}")
    
    if args.type == "data_scorer":
        data_score(args, args.input_data_path, args.output_data_path)
    else:
        raise ValueError(f"Invalid type: {args.type}")
