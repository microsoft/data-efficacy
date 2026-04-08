import os
import copy
#import wandb
import torch
from utils import get_model, get_tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate.utils import DistributedType
from transformers import TrainingArguments, Trainer
#from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding

os.environ["TOKENIZERS_PARALLELISM"] = "false"
IGNORE_INDEX = -100

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "shuffle": False,
            #"sampler": self._get_train_sampler(),
            #"worker_init_fn": seed_worker,
        }
        self.train_dataset = self._remove_unused_columns(self.train_dataset, description="training")
        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(**inputs)
        losses = outputs.loss
        return (losses, outputs) if return_outputs else losses

def preprocess(tokenizer, examples, instruct_field, text_field, max_length=1024, padding="max_length"):# max_length, longest
    instructs = examples.get(instruct_field, None)
    texts = examples.get(text_field, None)

    input_ids_list = list()
    labels_list = list()
    attention_mask_list = list()
    for i in range(len(texts)):
        instruct = instructs[i] if isinstance(instructs, list) else ""
        text = instruct + texts[i]

        tokenized_example = tokenizer(text, padding=padding, add_special_tokens=True, truncation=True, max_length=max_length)
        input_ids = tokenized_example["input_ids"]
        attention_mask = tokenized_example["attention_mask"]
        labels = copy.deepcopy(input_ids)

        tokenized_instruct = tokenizer(instruct, padding=padding, add_special_tokens=True, truncation=True, max_length=max_length)
        input_len = len(tokenized_instruct["input_ids"]) - tokenized_instruct["input_ids"].count(tokenizer.pad_token_id)
        labels[:input_len] = [IGNORE_INDEX] * input_len

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return dict(
            input_ids=input_ids_list,
            labels=labels_list,
            attention_mask=attention_mask_list,
        )

def train(args):
    # device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}, cpu count: {os.cpu_count()}")

    # tokenizer.
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer = get_tokenizer(args)

    # model.
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", trust_remote_code=True)
    model = get_model(args, device=device)
    model.train()

    # data.
    dataset = load_dataset("json", data_files={"train": args.data_path})
    train_dataset = dataset["train"]
    # For debug.
    #train_dataset = train_dataset.select(range(100))
    
    train_dataset = train_dataset.map(
        lambda example: preprocess(tokenizer, example, args.instruct_field, args.text_field, args.max_length), 
        batched=True, num_proc=os.cpu_count(), load_from_cache_file=False)
    #train_dataset = train_dataset.filter(lambda example: len(example["input_ids"]) > 0)
    #if args.shuffle_data:
    #    train_dataset = train_dataset.shuffle(seed=args.seed)

    # train args.
    #wandb.init(project=os.path.basename(args.save))
    training_args = TrainingArguments(
        output_dir=args.save, 
        overwrite_output_dir=True,
        fp16=True,
        deepspeed=args.deepspeed_config,
        lr_scheduler_kwargs={"min_lr": args.lr_min},
        #report_to="none",
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=10,
    )

    training_args.set_training(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    training_args.set_optimizer(
        name=args.optimizer_name, 
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        beta1=args.adam_beta,
        beta2=args.adam_beta2,
        epsilon=args.adam_eps,
        )

    training_args.set_lr_scheduler(
        name=args.lr_scheduler_type,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_iters,
    )

    training_args.set_logging(
        strategy=args.log_name,
        steps=args.log_interval,
        report_to=args.report_name,
        level=args.log_level,
    )

    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        #tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.save)
    tokenizer.save_pretrained(args.save)
    torch.distributed.destroy_process_group()
