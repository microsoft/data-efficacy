import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
from tqdm import tqdm
from utils import load_yaml, add_args, load_jsonl, write_jsonl
from data_scoring.kenlm.model import KenlmModel


def download_model(args):
    model_folder = os.path.join(args.model_path, args.language_type)

    print("Downloading KenLM model...")
    os.makedirs(model_folder, exist_ok=True)
    base_url = "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/"
    names = (
        "en.arpa.bin", 
        "en.sp.model", 
        "en.sp.vocab", 
    )
    for name in names:
        os.system(f"wget -O {os.path.join(model_folder, name)} {base_url + name}")
    print(f"KenLM model is downloaded to {model_folder}")


def scoring(args):
    model = KenlmModel.from_pretrained(args.model_path, args.language_type)
    items = load_jsonl(args.input_data_path)

    print(f"Scoring full dataset from {args.input_data_path} ...")
    for item in tqdm(items):
        if isinstance(args.text_fields, str):
            text_to_process = item[args.text_fields]
        elif isinstance(args.text_fields, list):
            text_to_process = " ".join([item[field] for field in args.text_fields if field in item])
        else:
            raise ValueError("args.text_fields name not exist.")
        item[args.score_field] = model.get_perplexity(text_to_process)

    write_jsonl(args.output_data_path, items)
    print(f"KenLM data scoring completed. The scored data is saved to {args.output_data_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of KenLM data scorer.") 
    parser.add_argument("--input-data-path", type=str, required=True, help="Path of input data.")
    parser.add_argument("--output-data-path", type=str, required=True, help="Path of output data.")
    parser.add_argument("--config-path", type=str, required=True, help="Path of config file.")

    args = parser.parse_args()
    process_param = load_yaml(args.config_path)
    args = add_args(args, process_param)
    
    download_model(args)
    scoring(args)
