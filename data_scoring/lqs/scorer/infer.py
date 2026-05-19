import os
import sys
sys.path.insert(0, os.getcwd())
import json
import torch
from tqdm import tqdm

from .modeling import DataScorerModel
from utils import get_model, get_tokenizer, load_jsonl, write_jsonl, BOS_MODELS

torch.backends.cudnn.enabled = False 

class DataScorerInfer():
    def __init__(self, args):
        self.args = args
        self.max_length = args.max_length
        self.device = torch.cuda.current_device()
        self.tokenizer = get_tokenizer(args)
        self.model = self.load_model(args)
    
    def load_model(self, args):
        with open(os.path.join(args.model_path, "config.json")) as f:
            config = json.load(f)
        bias = config.get("bias", False) or args.data_scorer_bias
        encoding = config.get("encoding", None) or args.data_scorer_encoding
        model = DataScorerModel(
            args, "cpu", os.path.join(os.getcwd(), config["base_model_path"].strip("/")), bias=bias, encoding=encoding)
        model.load_state_dict(torch.load(os.path.join(args.model_path, "data_scorer_model.pt"), map_location="cpu"))
        model = model.to(self.device)
        model.eval()
        if args.torch_compile is not None:
            model.inference = torch.compile(model.inference, mode=args.torch_compile)
        return model
    
    def inference(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        tokens = tokens[:self.max_length]

        input_ids = self.tokenizer.pad_token_id * torch.ones(1, self.max_length, dtype=torch.long, device=self.device)
        input_ids[0][:len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        attention_mask = torch.zeros(1, self.max_length, dtype=torch.long, device=self.device)
        attention_mask[0][:len(tokens)] = 1

        pos = torch.zeros(1, dtype=torch.long, device=self.device)
        pos[0] = len(tokens) - 1

        with torch.no_grad():
            score = self.model.inference(input_ids=input_ids, attention_mask=attention_mask, pos=pos)
            if self.args.torch_compile:
                score = score.clone()
            return score.item()


def data_score(args, input_data_path, output_data_path):
    data_scorer_infer = DataScorerInfer(args)
    items = load_jsonl(input_data_path)
    for item in tqdm(items):
        item["score"] = data_scorer_infer.inference(item["text"])
    write_jsonl(output_data_path, items)
