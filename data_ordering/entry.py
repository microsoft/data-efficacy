import argparse

try:
    from . import folding, saw, segment, shuffle, sorting, stair, zigzag
    from .common import add_config_args, ensure_permutation, load_jsonl, load_yaml, write_jsonl
except ImportError:
    import folding
    import saw
    import segment
    import shuffle
    import sorting
    import stair
    import zigzag
    from common import add_config_args, ensure_permutation, load_jsonl, load_yaml, write_jsonl


METHODS = {
    "shuffle": shuffle.order,
    "sorting": sorting.order,
    "folding": folding.order,
    "zigzag": zigzag.order,
    "segment": segment.order,
    "stair": stair.order,
    "saw": saw.order,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Order a scored JSONL dataset.")
    parser.add_argument("--input_data_path", required=True, type=str, help="Path to the input .jsonl file.")
    parser.add_argument("--output_data_path", required=True, type=str, help="Path to the output .jsonl file.")
    parser.add_argument("--method", choices=sorted(METHODS), default="folding", help="Data ordering method.")
    parser.add_argument("--config_path", default="data_ordering/config/folding.yaml", type=str, help="YAML config path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = add_config_args(args, load_yaml(args.config_path))

    print("Arguments received:")
    print(f"  input_data_path: {args.input_data_path}")
    print(f"  output_data_path: {args.output_data_path}")
    print(f"  method: {args.method}")
    print(f"  score_field: {args.score_field}")

    in_data = load_jsonl(args.input_data_path)
    out_data = METHODS[args.method](in_data, args)
    ensure_permutation(in_data, out_data)
    write_jsonl(args.output_data_path, out_data)
    print(f"Wrote {len(out_data)} records.")


if __name__ == "__main__":
    main()
