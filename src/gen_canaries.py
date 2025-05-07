import argparse
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

def jaccard_similarity(set1, set2):
    set1 = set(set1)
    set2 = set(set2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union

def is_duplicate(seq, canaries, threshold=0.2):
    
    for canary in canaries:
        if jaccard_similarity(seq, canary) > threshold:
            return True

    return False

def check_retokenization(seq, tokenizer):
    
    for token in seq:
        detokenized = tokenizer.decode(token)
        token_again = tokenizer(detokenized)['input_ids'][0]
        if token != token_again:
            print(token, detokenized, token_again)
            return False

    return True

def gen_canaries(model, tokenizer, target_tokenizer, args):

    canaries = []
    
    input = tokenizer([""] * args.batch_size, return_tensors="pt").to(args.device)

    total_samples = 0
    step = 0
    duplicates = 0
    retokenize_issue = 0
    eos_canaries = 0

    while len(canaries) < args.num_canaries:
        if step > 0 and step % 5 == 0:
            samples = len(canaries)
            print(
                f"Step: {step} | total: {total_samples} | accepted: {samples} | duplicates: {duplicates} | eos canaries: {eos_canaries} | retokenize issues: {retokenize_issue}")
            
        try:
            # max_length=args.seq_len+25 to get a bit more, which will get truncated anyway
            generated_ids = model.generate(
                    input["input_ids"],
                    max_length=args.seq_len + 25,
                    do_sample=True,
                    temperature=args.temp,
            )

            for idx in range(args.batch_size):

                total_samples += 1
                
                generated_canary_tokens = generated_ids[idx].detach().cpu().numpy()

                # make sure we don't have sequences with EOS token generated
                if tokenizer.eos_token_id in generated_canary_tokens: 
                    print(tokenizer.decode(generated_canary_tokens))
                    eos_canaries += 1
                    continue
                    
                # get the raw text, deleting the first token
                generated_canary_text = tokenizer.decode(generated_canary_tokens[1:])               
                
                # get them tokenized using the target tokenizer
                generated_canaries_target_tokens = target_tokenizer(generated_canary_text) 
                
                # truncate this to the right length
                if len(generated_canaries_target_tokens['input_ids']) < args.seq_len:
                    continue
                else:
                    generated_canaries_target_tokens_truncated = generated_canaries_target_tokens['input_ids'][:args.seq_len]

                # get the canary text
                new_canary_tokens = generated_canaries_target_tokens_truncated
                
                if is_duplicate(new_canary_tokens, canaries, threshold=args.jaccard_threshold):
                    duplicates += 1
                    continue
                
                # make sure the canary does not contain any weird tokens for the target tokenizer
                if not check_retokenization(new_canary_tokens, tokenizer=target_tokenizer):
                    retokenize_issue += 1
                    continue

                # make sure we do not add too many canaries
                if len(canaries) == args.num_canaries:
                    continue
                
                print("New canary selected: ", target_tokenizer.decode(generated_canaries_target_tokens_truncated))

                canaries.append(new_canary_tokens)

        except (RuntimeError, ValueError):
            # https://github.com/facebookresearch/llama/issues/380
            pass

        with open(args.output + ".tmp", "wb") as file:
            pickle.dump(canaries, file)

        step += 1

    return canaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path-to-model", type=str, required=True)
    parser.add_argument("--path-to-tokenizer", type=str, required=True)
    parser.add_argument("--path-to-target-tokenizer", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)

    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Target canary length (in number of tokens of the target tokenizer)",
    )
    parser.add_argument("-n", "--num-canaries", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--jaccard-threshold", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device)
    print("Using device:", device)

    random.seed(args.seed * args.seq_len)
    np.random.seed(args.seed * args.seq_len)
    torch.manual_seed(args.seed * args.seq_len)

    model = LlamaForCausalLM.from_pretrained(args.path_to_model, torch_dtype=torch.float32).to(device)  # type: ignore
    tokenizer = LlamaTokenizer.from_pretrained(args.path_to_tokenizer, torch_dtype=torch.float32)
    tokenizer.pad_token = tokenizer.eos_token

    target_tokenizer = AutoTokenizer.from_pretrained(args.path_to_target_tokenizer, torch_dtype=torch.float16)
    target_tokenizer.pad_token = target_tokenizer.eos_token

    canaries = gen_canaries(model, tokenizer, target_tokenizer, args)

    with open(args.output, "wb") as file:
        pickle.dump(canaries, file)

    print(f"Finished. Generated {len(canaries)} samples")
