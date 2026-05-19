import os
import sys
sys.path.insert(0, os.getcwd())
import json
import argparse
import multiprocessing
from data_scoring.lqs.data_utils import DistributedMMapIndexedDataset 

from utils import add_args, load_yaml, get_tokenizer, load_yaml, add_args
from data_scoring.lqs.tools.get_name_lqs import s1_1_full_target_token_model, s1_2_full_token_data


data = None
tokenizer = None

def worker_init(bin_dir, token_model):
    global data, tokenizer
    data = DistributedMMapIndexedDataset(bin_dir, "data")
    tokenizer = token_model


def process_doc(doc_id):
    global data, tokenizer
    tokens = data[doc_id]
    text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
    line = json.dumps(
        {"id": doc_id, "input": "", "text": text},
        ensure_ascii=False
    )
    return line


def decode_bin_to_jsonl_fast(bin_data_path, jsonl_output_path, tokenizer_model, num_workers=32):
    dataset = DistributedMMapIndexedDataset(bin_data_path, "data")
    total = len(dataset)
    if total == 0:
        print("No data found in the specified directory.")
        return
    print(f"Found {total} entries. Starting parallel decoding with {num_workers} workers...")

    pool = multiprocessing.Pool(
        processes=num_workers,
        initializer=worker_init,
        initargs=(bin_data_path, tokenizer_model)
        )

    with open(jsonl_output_path, "w", encoding="utf-8") as fout:
        for doc_id, json_line in enumerate(pool.imap(process_doc, range(total), chunksize=100)):
            fout.write(json_line + "\n")
            if doc_id and doc_id % 10000 == 0:
                print(f"Decoded {doc_id}/{total} documents...")

    pool.close()
    pool.join()
    print(f"Decoding complete. All data saved to {jsonl_output_path}")


def decode_bin_to_jsonl(bin_data_path, jsonl_output_path, tokenizer):
    data = DistributedMMapIndexedDataset(bin_data_path, "data")
    if len(data) == 0:
        print("No data found in the specified directory.")
        return

    print(f"Found {len(data)} entries in the dataset. Decoding to {jsonl_output_path}...")

    with open(jsonl_output_path, "w") as jsonl_file:
        for doc_id, tokens in enumerate(data):
           
            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

            json_line = {
                "id": doc_id,
                "text": text
            }
            jsonl_file.write(json.dumps(json_line, ensure_ascii=False) + "\n")

            if doc_id % 10000 == 0:
                print(f"Decoded {doc_id}/{len(data)} documents...")

    print(f"Decoding complete. All data saved to {jsonl_output_path}")


def main(args):
    tokenizer = get_tokenizer(args, model_path=args.jsonl_tokenizer_path, model_type=args.jsonl_model_type)

    # decode_bin_to_jsonl(
    #     bin_data_path=args.bin_data_path,
    #     jsonl_output_path=args.jsonl_output_path,
    #     tokenizer=tokenizer
    # )

    decode_bin_to_jsonl_fast(
        bin_data_path=args.bin_data_path,
        jsonl_output_path=args.jsonl_output_path,
        tokenizer_model=tokenizer
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert bin file to json format for token data.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data_training", "scorer_data_infer"], default="scorer_data_infer", help="The running steps of LQS.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    process_param = load_yaml(args.config_path)
    args = add_args(args, process_param, args.lqs_process)

    if args.lqs_process == "scorer_data_infer":
        args.bin_data_path = os.path.join(args.bin_data_path, s1_2_full_token_data(process_param))
        args.jsonl_output_path = args.bin_data_path + '.jsonl'
        args.jsonl_tokenizer_path = os.path.join(args.jsonl_tokenizer_path, s1_1_full_target_token_model(process_param))
    else:
        raise ValueError(f"Error lqs process for this file: {args.lqs_process}")    
    
    main(args)
