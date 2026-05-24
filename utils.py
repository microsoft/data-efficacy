import os
import yaml
import json
import time
import random
import argparse
import numpy as np
from datetime import timedelta
from numerize.numerize import numerize

import torch
import torch.distributed as dist
from torch.distributed import get_rank
import deepspeed
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

try:
    from transformers import mpu
except:
    mpu = None

WANDB_PROJ_NAME = "DELT"
PAD_EOS_MODELS = ["gpt2", "opt", "llama", "mistral"]
BOS_MODELS = ["fairseq", "mistral", "llama"]
#HF_TOKEN = os.environ["HF_TOKEN"]
HF_TOKEN = None


# Logging
def print_args(args):
    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank(log_str, save_path, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)


# Distributed
def all_gather(t, dim=0, world_size=None, group=None, op="cat"):
    if world_size is None:
        world_size = dist.get_world_size()
    all_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(all_t, t, group=group)
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    return all_t


# Initialize
def set_random_seed(seed, mp=False):
    """Set random seed for reproducability."""
    if dist.is_initialized():
        seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if mp:
        #     mpu.model_parallel_cuda_manual_seed(seed)


def add_args(args, method_params, fields=None):
    if isinstance(fields, str):
        fields = [fields]
    if fields is None:
        fields = method_params.keys()

    for field in fields:
        if field in method_params:
            value = method_params[field]
            if isinstance(value, dict):
                for key, sub_value in value.items():
                    if not hasattr(args, key) or getattr(args, key) is None:
                        setattr(args, key, sub_value)
            else:
                if not hasattr(args, field) or getattr(args, field) is None:
                    setattr(args, field, value)
    return args


def base_training_hp_suffix(args):
    suffix = ""
    suffix += (f"e{args.epochs}" if args.epochs is not None else f"t{numerize(args.total_iters)}") + \
        (f"-w{numerize(args.warmup_iters)}" if args.warmup_iters > 0 else "") + \
        (f"-bs{args.batch_size}-lr{args.lr}{args.scheduler_name}-G{args.gradient_accumulation_steps}") + \
        (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "")
    return suffix


def base_infer_hp_suffix(args):
    return ""


def base_model_suffix(args):
    return f"{args.ckpt_name.replace('/', '_')}"


def base_data_suffix(args):
    return f"{args.data_name.replace('/', '_')}"


def init_distributed(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=300))


def init_distributed_ds(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    deepspeed.init_distributed(timeout=timedelta(minutes=300))


def init_deepspeed(args):
    if args.deepspeed_config is not None:
        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)

        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_config["gradient_clipping"] = args.clip_grad
        ds_config["steps_per_print"] = 10000000
        
        if not args.do_train:
            ds_config["zero_optimization"]["stage"] = 0
        
        if not ds_config["fp16"]["enabled"]:
            args.fp32 = True
        
        args.deepspeed_config = None
    else:
        ds_config = None
    
    return ds_config


def init_deepspeed_infer(args):
    if args.deepspeed_config is not None:
        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)

        ds_config["zero_optimization"]["stage"] = 0
        
        if not ds_config["fp16"]["enabled"]:
            args.fp32 = True
        
        args.deepspeed_config = None
    else:
        ds_config = None
    
    return ds_config


def init_env(seed):
    torch.set_num_threads(16)
    torch.backends.cudnn.enabled = False

    print('Random Seed: ', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if os.environ.get('DETERMINISTIC') is not None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
 
    # be consistent with nanogpt settings
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.cuda.manual_seed_all(seed)
    print('Set Random Seed Successful: ', seed)


def init(args, do_distributed=True):
    # init distributed
    if do_distributed:
        if args.deepspeed:
            init_distributed_ds(args)
        else:
            init_distributed(args)

    if args.model_parallel:
        assert dist.get_world_size() % args.model_parallel_size == 0 
        mpu.initialize_model_parallel(args.model_parallel_size)

    set_random_seed(args.seed, args.model_parallel)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.time_stamp = cur_time

    init_env(args.seed)


# Load and save model
def get_model(args, device, model_path=None, config=None, from_scratch=None, model_cls=None):
    if model_path is None:
        model_path = args.model_path
    print_rank("Initializing model from {}".format(model_path), rank=0)
    print_rank(f"Attention Implementation: {args.attn_impl}")
    if config is None:
        config = AutoConfig.from_pretrained(model_path, attn_implementation=args.attn_impl)
        
    if args.dropout_path_rate is not None:
        config.drop_path_rate = args.dropout_path_rate
    if args.xops_attn:
        assert args.attn_impl == "eager"
        print_rank("Xops Attention")
        config.use_memory_efficient_attention = True
    else:
        config.use_memory_efficient_attention = False

    if args.model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():
            model = parallel_model_map[args.model_type].half()
        load_parallel(model, args.model_path)

        if mpu.get_data_parallel_rank() == 0:
            print(' > number of parameters on model parallel rank {}: {}'.format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()])), flush=True)
    else:
        config.is_model_parallel = False
        from_scratch = from_scratch if from_scratch is not None else args.from_scratch
        model_cls = model_cls if model_cls is not None else AutoModelForCausalLM
        if from_scratch:
            print('Pre-train mode: train from scratch ...')
            model = model_cls.from_config(config, attn_implementation=args.attn_impl).to(device)
        else:
            print('Fine-tune mode: load pre-train model from: ', str(model_path))
            dtype = torch.float32 if args.fp32 else torch.float16
            model = model_cls.from_pretrained(model_path, config=config, device_map={"": device}, torch_dtype=dtype)
            # model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", trust_remote_code=True)

        #if dist.get_rank() == 0:
        #    print(' > number of parameters: {}'.format(
        #        sum([p.nelement() for p in model.parameters()])), flush=True)
        # model = DDP(model)
        # NOTE: no need for DDP since deepspeed has done
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model


def get_tokenizer(args, model_path=None, model_type=None):
    if model_path is None:
        model_path = args.model_path
    
    if model_type is None:
        model_type = args.model_type

    if args.max_length is None or not args.truncation:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=args.max_length,
            truncation=args.truncation,
            #padding_side=args.padding_side,
            #use_fast=False,
        )
    
    if model_type in PAD_EOS_MODELS:
        #tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def load_parallel(model, load_dir):
    mp_rank = mpu.get_model_parallel_rank()
    assert mpu.get_model_parallel_world_size() != 1
    checkpoint_name = os.path.join(load_dir, f"mp{mpu.get_model_parallel_world_size()}", f"pytorch_model_{mp_rank}.bin")
    assert os.path.exists(checkpoint_name), f"{checkpoint_name} does not exist."
    model = load_checkpoint_and_dispatch(model=model, checkpoint=checkpoint_name, device_map={"": torch.cuda.current_device()}, dtype=torch.float16)
    dist.barrier()
    print(f"Rank {get_rank()}: {checkpoint_name} loaded.")


def save_parallel(model, save_dir):
    mp_rank = mpu.get_model_parallel_rank()
    os.makedirs(os.path.join(save_dir, f"mp{mpu.get_model_parallel_world_size()}"), exist_ok=True)
    checkpoint_name = os.path.join(save_dir, f"mp{mpu.get_model_parallel_world_size()}", f"pytorch_model_{mp_rank}.bin")
    torch.save(model.state_dict(), checkpoint_name)
    print(f"Rank {get_rank()}: {checkpoint_name} saved.")


def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def write_yaml(file_path, data):
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    return data


def write_jsonl(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        for entry in data:
            json_line = json.dumps(entry)
            file.write(json_line + "\n")


def download_repo(repo_id, save_dir, repo_type="model", revision=None, token=None, allow_patterns=None, ignore_patterns=None):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=save_dir,
        local_dir_use_symlinks=False, 
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        token=HF_TOKEN,
    )
    print(f"Repository '{repo_id}' ({repo_type}) downloaded to: {save_dir}")


def download_model(model_id, save_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(save_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.save_pretrained(save_dir, safe_serialization=False)
    print(f"Model '{model_id}' has been saved to '{save_dir}'.")


def download_data(dataset_id, name, save_dir, split_name=None, sample_size=-1):
    dataset = load_dataset(
        dataset_id, 
        name=name, 
        split=(split_name if split_name != "" else None), 
        streaming=False, 
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    
    sampled_data = []
    for i, example in enumerate(dataset):
        if sample_size > 0 and i >= sample_size:
            break
        sampled_data.append(example)
    
    sampled_dataset = Dataset.from_list(sampled_data)
    sampled_dataset.to_json(save_dir)
    
    print(f"Dataset '{dataset_id}' has been saved to '{save_dir}' with {len(sampled_data)} samples.")


def download_redpajama_samples(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base_url = "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/"
    names = (
        "cc_2019-30_sample.jsonl", 
        "cc_2020-05_sample.jsonl", 
        "cc_2021-04_sample.jsonl", 
        "cc_2022-05_sample.jsonl", 
        "cc_2023-06_sample.jsonl"
    )
    for name in names:
        os.system(f"wget -O - {base_url + name} >> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF dataset or model.")
    parser.add_argument("--content", type=str, required=True, choices=["dataset", "model"], default="dataset", help="The content to be downloaded.")
    parser.add_argument("--id", type=str, required=True, help="Input dataset id or model id.")
    parser.add_argument("--data-name", type=str, required=False, default=None, help="Split name of dataset.")
    parser.add_argument("--save-dir", type=str, required=True, help="Output path of saved dataset or model.")
    parser.add_argument("--split-name", type=str, required=False, default=None, help="Split name of dataset.")
    parser.add_argument("--sample-size", type=int, required=False, default=-1, help="Sample size of dataset.")

    args = parser.parse_args()

    if args.content == "model":
        download_model(args.id, args.save_dir)

    if args.content == "dataset":
        if args.id == "togethercomputer/RedPajama-Data-1T-Sample":
            download_redpajama_samples(args.save_dir)
        else:
            download_data(args.id, args.data_name, args.save_dir, args.split_name, args.sample_size)

    if args.content == "repo":
        download_repo(args.id, args.save_dir)
