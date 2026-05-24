import ast
import json
import os
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    yaml = None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def load_yaml(file_path: str) -> Mapping[str, Any]:
    """Load the flat YAML configs used by the ordering scripts.

    PyYAML is preferred when available. The small fallback keeps
    `data_ordering` runnable in a fresh Python environment because the module
    only needs simple `key: value` config files.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        if yaml is not None:
            data = yaml.safe_load(file)
            return data or {}

        data = {}
        for raw_line in file:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = _parse_scalar(value)
        return data


def load_jsonl(file_path: str) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(file_path: str, data: Iterable[Mapping[str, Any]]) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def add_config_args(args: Any, method_params: Mapping[str, Any]) -> Any:
    for field, value in method_params.items():
        if not hasattr(args, field) or getattr(args, field) is None:
            setattr(args, field, value)
    return args


def get_score(item: Mapping[str, Any], score_field: str) -> float:
    if score_field not in item:
        raise KeyError(f"Missing score field '{score_field}' in item: {item}")
    return float(item[score_field])


def sorted_indices(data: Sequence[Mapping[str, Any]], score_field: str, ascending: bool = True) -> List[int]:
    return sorted(
        range(len(data)),
        key=lambda idx: get_score(data[idx], score_field),
        reverse=not ascending,
    )


def sort_data(data: Sequence[Mapping[str, Any]], score_field: str, ascending: bool = True) -> List[dict]:
    return [data[idx] for idx in sorted_indices(data, score_field, ascending)]


def gumbel_rank_jitter(indices: Sequence[int], tau: float = 1.0, use_gumbel: bool = False, seed: int = 42) -> List[int]:
    if not use_gumbel:
        return list(indices)

    rng = np.random.default_rng(seed)
    noise = rng.gumbel(size=len(indices)) * tau
    perturbed_positions = np.arange(len(indices), dtype=float) + noise
    order = np.argsort(perturbed_positions, kind="stable")
    idx = np.asarray(indices)
    return list(idx[order])


def window_based_shuffle(data: Sequence[dict], window_size: int = 0, seed: int = 42) -> List[dict]:
    if window_size is None or window_size <= 1:
        return list(data)

    rng = np.random.default_rng(seed)
    shuffled_data: List[dict] = []
    for start in range(0, len(data), window_size):
        chunk = list(data[start : start + window_size])
        rng.shuffle(chunk)
        shuffled_data.extend(chunk)
    return shuffled_data


def validate_layers(layers: int, name: str) -> int:
    layers = int(layers)
    if layers < 1:
        raise ValueError(f"{name} must be >= 1, got {layers}")
    return layers


def folding_order(sorted_data: Sequence[dict], layers: int) -> List[dict]:
    layers = validate_layers(layers, "folding_layer")
    ordered: List[dict] = []
    for layer in range(layers):
        ordered.extend(sorted_data[layer::layers])
    return ordered


def zigzag_order(sorted_data: Sequence[dict], layers: int) -> List[dict]:
    layers = validate_layers(layers, "zigzag_layer")
    ordered: List[dict] = []
    for layer in range(layers):
        layer_data = list(sorted_data[layer::layers])
        if layer % 2 == 1:
            layer_data.reverse()
        ordered.extend(layer_data)
    return ordered


def cross_guidance_order(
    sorted_data: Sequence[dict],
    num_sections: int,
    transition_ratio: float,
    folding_layer: int,
    mode: str,
) -> List[dict]:
    """Apply STR/SAW transition regions on top of globally sorted data."""
    num_sections = validate_layers(num_sections, "num_sections")
    folding_layer = validate_layers(folding_layer, "folding_layer")
    if transition_ratio < 0:
        raise ValueError(f"folding_ratio must be >= 0, got {transition_ratio}")
    if mode not in {"folding", "zigzag"}:
        raise ValueError(f"Unsupported transition mode: {mode}")

    n_items = len(sorted_data)
    if n_items == 0 or num_sections == 1 or transition_ratio == 0:
        return list(sorted_data)

    split_points = [round(n_items * section / num_sections) for section in range(1, num_sections)]
    radius = round(n_items * transition_ratio)
    ordered: List[dict] = []
    cursor = 0

    for split_point in split_points:
        transition_start = max(cursor, split_point - radius)
        transition_end = min(n_items, split_point + radius)

        if cursor < transition_start:
            ordered.extend(sorted_data[cursor:transition_start])

        transition = list(sorted_data[transition_start:transition_end])
        if mode == "folding":
            ordered.extend(folding_order(transition, folding_layer))
        else:
            ordered.extend(zigzag_order(transition, folding_layer))
        cursor = transition_end

    if cursor < n_items:
        ordered.extend(sorted_data[cursor:n_items])

    return ordered


def ensure_permutation(input_data: Sequence[dict], output_data: Sequence[dict]) -> None:
    if len(input_data) != len(output_data):
        raise ValueError(f"Ordering changed data size from {len(input_data)} to {len(output_data)}")
