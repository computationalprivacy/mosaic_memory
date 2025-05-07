import os
import pickle
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from utils import compute_perplexity, ratio_auc
import argparse

EVAL_DS_PATH = "SOME_DATA_DIR/books_eval_100"
CACHE_DIR = "SOME_DATA_DIR/cache/huggingface"
MAX_CONTEXT_SIZE = 2048


def split_into_chunks(examples, seq_len=MAX_CONTEXT_SIZE):
    ret = []
    for example in examples["input_ids"]:
        ret.extend([example[i: i + seq_len] for i in range(0, len(example), seq_len)])
    return {"input_ids": ret}

def load_model(model_name, device, reduced_precision):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    if reduced_precision:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=CACHE_DIR)

    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)

    return model, tokenizer


def get_MIA_data(path_to_members, path_to_non_members, tokenizer):
    MIA_data = {}

    with open(path_to_members, 'rb') as f:
        member_tokens = pickle.load(f)
    MIA_data["members"] = tokenizer.batch_decode(member_tokens)

    with open(path_to_non_members, 'rb') as f:
        non_member_tokens = pickle.load(f)
    MIA_data["non_members"] = tokenizer.batch_decode(non_member_tokens)

    return MIA_data


def prepare_eval_ds(tokenizer):
    eval_ds = load_from_disk(EVAL_DS_PATH).train_test_split(test_size=0.01, seed=43)["test"]
    eval_ds = eval_ds.map(lambda example: tokenizer(example["text"]), batched=True, num_proc=10)
    eval_ds = eval_ds.map(split_into_chunks, batched=True, remove_columns=eval_ds.column_names)
    return eval_ds


def main(args):
    print("Learning rate: ", args.learning_rate_e6)
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print(f"Loading target model {args.model}...")
    model, tokenizer = load_model(args.model, args.device, args.reduced_precision)
    
    canary_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=CACHE_DIR)

    # load training data
    print("Loading the training data...")
    training_data = load_from_disk(args.training_data)
    print("Tokenizing the training data...")
    training_data = training_data.map(lambda example: tokenizer(example["text"]),
                                      batched=True, num_proc=10)
    training_data = training_data.map(split_into_chunks, remove_columns=training_data.column_names,
                                      batched=True, num_proc=10)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # get some eval dataset as well
    print("Loading and prepping the eval data...")
    eval_ds = prepare_eval_ds(tokenizer)

    # get the MIA data
    print("Loading the canaries and non-members..")
    MIA_data = get_MIA_data(args.path_to_members, args.path_to_non_members, canary_tokenizer)

    target_tokens = {}

    for k, v in MIA_data.items():
        target_tokens[k] = tokenizer.batch_encode_plus(
            list(v), return_tensors="pt", padding="longest", add_special_tokens=False
        ).to(args.device)

    def custom_metrics(p: EvalPrediction):
        r = {}
        target_ppl = {}

        try:
            for k, v in target_tokens.items():
                target_ppl[k] = compute_perplexity(
                    model,
                    v.input_ids,
                    v.attention_mask,
                )

            for k in target_ppl:
                r[f"target_ppl_{k}"] = target_ppl[k].mean()

            r["auc"] = ratio_auc(target_ppl["members"], target_ppl["non_members"])
        except:
            pass
        
        return r

    training_data = training_data.shuffle(seed=42)
    model_name = args.model.replace("/", "_")
    run_name = model_name + "_" + args.run_name

    training_args = TrainingArguments(
        output_dir=f"SOME_DATA_DIR/.output/{model_name}_{run_name}",
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=50,
        learning_rate=args.learning_rate_e6 * (10 ** (-6)),
        weight_decay=0.01,
        push_to_hub=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*2,
        gradient_accumulation_steps=args.accumulate_steps,
        # save_steps=1000,
        save_strategy="no",
        save_total_limit=1,
        # include_tokens_per_second=True,
        report_to=["wandb"],
        # eval_accumulation_steps=4,
        logging_first_step=True,
        num_train_epochs=1,
        run_name=run_name,
        bf16=args.reduced_precision,
        #fp16_full_eval=args.reduced_precision,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=custom_metrics,
        preprocess_logits_for_metrics=lambda x, y: torch.empty(0),
    )

    trainer.evaluate()
    trainer.train()
    trainer.evaluate()

    if args.save_checkpoint:
        print("Training is done - saving checkpoint")

        save_path = f"SOME_DATA_DIR/model_checkpoints/{model_name}_checkpoints/{run_name}"
        model.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--path_to_members", type=str, required=True)
    parser.add_argument("--path_to_non_members", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="finetune_mosaic_near_dupl")
    parser.add_argument("--learning_rate_e6", type=float, default=3)
    parser.add_argument("--save_checkpoint", type=str, default=1)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--reduced-precision", action="store_true")
    args = parser.parse_args()
    main(args)
