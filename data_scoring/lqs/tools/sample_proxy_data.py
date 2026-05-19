import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
from tqdm import tqdm

from utils import add_args, load_yaml
from data_scoring.lqs.tools.get_name_lqs import s3_proxy_token_data, s1_2_full_token_data
from data_scoring.lqs.data_utils import DistributedMMapIndexedDataset, ChunkedDatasetBuilder


def main(args, output_path):
    
    np.random.seed(args.seed)
    os.makedirs(output_path, exist_ok=True)
        
    data = DistributedMMapIndexedDataset(args.data_path, "data", min_state=args.min_state, max_state=args.max_state)
    dtype = data[0].dtype.type
    builder = ChunkedDatasetBuilder(os.getcwd(), output_path, dtype)
    
    data_num = len(data)
    proxy_num = min(args.proxy_num, data_num)
    
    all_indices = set()
    for _ in tqdm(range(proxy_num)):
        idx = np.random.randint(data_num)
        while idx in all_indices:
            idx = np.random.randint(data_num)
        all_indices.add(idx)
    
    all_indices = list(all_indices)
    all_indices = sorted(all_indices)
    print("First 10 indices", list(all_indices)[:10])

    for idx in tqdm(all_indices):
        builder.add_np_item(data[idx])
        
    builder.finalize()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample proxy data for annotation.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data_training", "scorer_data_infer"], default="proxy_data", help="The running steps of LQS.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)
    process_param = load_yaml(args.config_path)

    if args.lqs_process == "proxy_data":
        output_path = os.path.join(args.save, s3_proxy_token_data(process_param))
        args.data_path = os.path.join(args.data_path, s1_2_full_token_data(process_param))
    else:
        raise ValueError(f"Error lqs process for this file: {args.lqs_process}")   
    
    main(args, output_path)
