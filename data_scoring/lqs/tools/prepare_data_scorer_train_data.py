import os
import sys
sys.path.insert(0, os.getcwd())
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from data_scoring.lqs.tools.get_name_lqs import s1_1_full_target_token_model, s4_3_prepare_scorer_data, s4_2_prepare_scorer_model, s4_1_annotated_proxy_token_data, s3_proxy_token_data
from utils import BOS_MODELS, get_tokenizer, load_yaml, add_args
from data_scoring.lqs.data_utils import DistributedMMapIndexedDataset, ChunkedDatasetBuilder, best_fitting_dtype


def normalize(scores):
    scores = scores - np.mean(scores)
    scores = scores / np.std(scores)
    scores = np.clip(scores, -3, 3)
    return scores


def main(args, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    src_tokenizer = get_tokenizer(
        args, model_path=args.model_path, model_type=args.model_type)
    dtype = best_fitting_dtype(src_tokenizer.vocab_size)

    dst_tokenizer = get_tokenizer(
        args, model_path=args.data_scorer_tokenizer_path, model_type=args.data_scorer_model_type)

    data_bin = DistributedMMapIndexedDataset(args.proxy_data_dir, "data", do_probe=True)
    data = []
    data_num = min(args.proxy_num, len(data_bin)) if args.proxy_num is not None else len(data_bin)
    for i in tqdm(range(data_num)):
        data.append(data_bin[i].astype(int))

    scores = torch.load(os.path.join(args.proxy_score_path, "grad_gamma.pt"), map_location="cpu").cpu().numpy()
    scores = normalize(scores)
    
    all_data = {
        "valid": (data[:args.proxy_dev_num], scores[:args.proxy_dev_num]),
        "train": (data[args.proxy_dev_num:], scores[args.proxy_dev_num:])
    }

    max_length_no_trunc = 0
    min_length_no_trunc = 1000000
    mean_length = 0

    for split in ["valid", "train"]:
        builder = ChunkedDatasetBuilder(os.getcwd(), output_path, dtype, split=split)
        x, y = all_data[split]
        new_y = []
        for lid, (xx, yy) in enumerate(zip(tqdm(x), y)):
            new_y.append(yy)
            eos_poses = np.where(xx == src_tokenizer.eos_token_id)[0]
            start = 0
            split_xx = []
            for p in eos_poses:
                split_xx.append(xx[start:p])
                start = p + 1
            split_xx.append(xx[start:])
            tokens = []
            for sxx in split_xx:
                s = src_tokenizer.decode(sxx, skip_special_tokens=True)
                _tokens = dst_tokenizer.encode(s, add_special_tokens=False)
                tokens.extend(_tokens)
                tokens.append(dst_tokenizer.eos_token_id)
            tokens.pop() # pop the last eos_token_id
            max_length_no_trunc = max(max_length_no_trunc, len(tokens))
            min_length_no_trunc = min(min_length_no_trunc, len(tokens))
            
            if args.data_scorer_model_type in BOS_MODELS:
                tokens = [dst_tokenizer.bos_token_id] + tokens[:args.max_length-1]

            if lid == 0:
                print(src_tokenizer.decode(xx))
                print(tokens)
                print(dst_tokenizer.decode(tokens))
            assert len(tokens) <= args.max_length
            mean_length += len(tokens)

            builder.add_np_item(np.array(tokens, dtype=dtype))
        
        builder.finalize()

        mean_length /= len(x)
        
        print(f"{split} max_length (before trunc): {max_length_no_trunc}, min_length: {min_length_no_trunc}, mean_length: {mean_length}")
        
        new_y = np.array(new_y)
        plt.plot(np.sort(new_y)[::-1], label="scored")
        baseline = np.ones_like(new_y) / len(new_y)
        plt.plot(baseline, label="baseline")
        plt.legend()
        plt.savefig(os.path.join(output_path, f"{split}_scores.png"))
        plt.close()

        with h5py.File(os.path.join(output_path, f"{split}_scores.hdf5"), "w") as f:
            f.create_dataset("scores", data=new_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare training data for data scorer.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data_training", "scorer_data_infer"], default="annotation_data", help="The running steps of LQS.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    process_param = load_yaml(args.config_path)
    args = add_args(args, process_param, args.lqs_process)

    if args.lqs_process == "annotation_data":
        output_path = os.path.join(args.proxy_save, s4_3_prepare_scorer_data(process_param))
        args.model_path = os.path.join(args.model_path, s1_1_full_target_token_model(process_param))
        args.data_scorer_tokenizer_path = os.path.join(args.data_scorer_tokenizer_path, s4_2_prepare_scorer_model(process_param))
        args.proxy_data_dir = os.path.join(args.proxy_data_dir, s3_proxy_token_data(process_param))
        args.proxy_score_path = os.path.join(args.proxy_score_path, s4_1_annotated_proxy_token_data(process_param))
    else:
        raise ValueError(f"Error lqs process for this file: {args.lqs_process}")    

    main(args, output_path)
