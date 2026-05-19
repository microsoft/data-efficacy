import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
from utils import load_yaml, add_args, download_model, download_data, download_repo
from data_scoring.lqs.tools.get_name_lqs import s1_1_full_target_token_model, s2_1_target_data, s4_2_prepare_scorer_model, s5_scorer_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF dataset or model.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data_training", "scorer_data_infer"], default="full_data", help="The running steps of LQS.")
    parser.add_argument("--content", type=str, required=True, help="Input dataset id or model id.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    process_param = load_yaml(args.config_path)
    args = add_args(args, process_param, args.lqs_process)

    if args.lqs_process == "full_data":
        output_model_path = os.path.join(args.output_model_path, s1_1_full_target_token_model(process_param))
    elif args.lqs_process == "target_data":
        output_data_path = os.path.join(args.output_data_path, s2_1_target_data(process_param, file_name=True))
    elif args.lqs_process == "annotation_data":
        output_model_path = os.path.join(args.output_model_path, s4_2_prepare_scorer_model(process_param))
    elif args.lqs_process == "checkpoint_download":
        output_model_path = os.path.join(args.output_model_path, s5_scorer_model(process_param))

    if args.content == "model":
        download_model(args.hf_model_id, output_model_path)

    if args.content == "dataset":
        download_data(args.hf_data_id, args.hf_data_name, output_data_path, args.split_name, args.sample_size)

    if args.content == "repo":
        download_repo(args.hf_model_id, output_model_path)
