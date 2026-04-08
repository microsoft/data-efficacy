import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import shuffle
import sorting
import folding
import zigzag
import segment
import stair
import saw
from utils import load_yaml, load_jsonl, add_args, write_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data ordering.")
    parser.add_argument("--input_data_path", type=str, help="Path to the input .jsonl file.")
    parser.add_argument("--output_data_path", type=str, help="Path to the output .jsonl file.")
    parser.add_argument("--method", type=str, choices=["shuffle", "sorting", "folding", "zigzag", "segment", "stair", "saw"], default="folding",
                        help="Ordering method: 'shuffle', 'sorting', and 'folding','zigzag','segment','stair','saw'. Defaults to 'folding'.")
    parser.add_argument("--config_path", type=str, default="./config/folding.yaml", help="Config file for additional parameters (YAML format).")

    args = parser.parse_args()

    args = add_args(args, load_yaml(args.config_path))

    print(f"  Arguments received:")
    print(f"  Input data path: {args.input_data_path}")
    print(f"  Selection method: {args.method}")
    print(f"  Score field: {args.score_field}")

    in_data = load_jsonl(args.input_data_path)
    if args.method == "shuffle":
        out_data = shuffle.order(in_data, args)
        print(f"  Random seed: {args.seed}")

    if args.method == "sorting":
        out_data = sorting.order(in_data, args)
        print(f"  Ascending: {args.ascending}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Use gumbel: {args.use_gumbel}")
        print(f"  Window size: {args.window_size}")

    if args.method == "folding":
        out_data = folding.order(in_data, args)
        print(f"  Folding layer: {args.folding_layer}")
        print(f"  Window size: {args.window_size}")

    if args.method == "zigzag":
        out_data = zigzag.order(in_data, args)
        print(f"  Zigzag layer: {args.zigzag_layer}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Use gumbel: {args.use_gumbel}")
        print(f"  Window size: {args.window_size}")

    if args.method == "segment":
        out_data = segment.order(in_data, args)
        print(f"  Front percentage: {args.x_pct}%")
        print(f"  Back percentage: {args.y_pct}%")
        print(f"  Front is high: {args.front_is_high}")
        print(f"  Back is high: {args.back_is_high}")
        if hasattr(args, 'seed'):
            print(f"  Random seed: {args.seed}")

    if args.method == "stair":
        out_data = stair.order(in_data, args)
        print(f"   Global Ascending: {args.ascending}")
        print(f"   Num sections: {args.num_sections}")
        print(f"   Folding ratio: {args.folding_ratio}")
        print(f"   Folding layer (in section): {args.folding_layer}")
        print(f"   Window size: {args.window_size}")

    if args.method == "saw":
        out_data = saw.order(in_data, args)
        print(f"   Global Ascending: {args.ascending}")
        print(f"   Num sections: {args.num_sections}")
        print(f"   Folding ratio: {args.folding_ratio}")
        print(f"   Folding layer (in section): {args.folding_layer}")
        print(f"   Window size: {args.window_size}")


    write_jsonl(args.output_data_path, out_data)
