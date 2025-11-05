import argparse
import pickle
import random
import re
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import os

SYSTEM_PROMPT = (
        "You are an assistant tasked with rephrasing the provided text in 9 different ways. "
        "Keep the original meaning intact (including the original natural or code language), but "
        "rephrase each version as if you are replacing the sentence entirely. "
        "Number the rephrased sequences using 1. to 9. and separate each by '---', like this: "
        "'1. rephrase 1 --- 2. rephrase 2 --- .. --- 9. rephrase 9'.")

OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def paraphrase_sequence(seq, model, tokenizer, terminators):

    user_prompt = f"Can you rephrase the following sequence? '{seq}'"

    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
    ]
        
    succeeded = False
    length = 1024
    
    while not succeeded:

        if model == 'gpt-4o':
            response = OPENAI_CLIENT.chat.completions.create(model=model, messages=messages, temperature=1.0, top_p=1.0, max_tokens=length)
            decoded = response.choices[0].message.content
            
        else:
            input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                    input_ids,
                    max_new_tokens=length,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
            )

            response = outputs[0][input_ids.shape[-1]:]
            decoded = tokenizer.decode(response, skip_special_tokens=True)
        
        # remove everything before 1.
        decoded = re.sub(r'^[^1]*1\.', '1.', decoded)
        
        # first try to see if we can split by '---' (ie whether it listened to instructions)
        rephrases = [s.strip() for s in re.split(r'\s*---\s*', decoded) if s.strip()]
        rephrases = [re.sub(r'^\d+\.\s*', '', r) for r in rephrases]  # Remove leading numbers

        if len(rephrases) == 9:
            succeeded = True
        else:
            # now we can still try if we can split just on the numbers
            parts = re.split(r'\s+(?:[1-9])\.(?!\d)\s*', decoded)
            rephrases = [s.strip() for s in parts if s.strip()]
            # remove '---' if present in any rephrase
            rephrases = [re.sub(r'\s*---\s*', '', r) for r in rephrases]
            # Remove leading numbers
            rephrases = [re.sub(r'^\d+\.\s*', '', r) for r in rephrases]
            
            if len(rephrases) == 9:
                succeeded = True
            else:
                print("Failed: ")
                print(decoded)
                
                if '9.' not in decoded:
                    length += 256
                    print("Increasing length to", length)
                
                continue

    return rephrases

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--paraphrase-model", type=str, required=True)
    parser.add_argument("--target-tokenizer", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    
    
    # load the paraphrase model
    if args.paraphrase_model == "gpt-4o":
        model = 'gpt-4o'
        tokenizer = None
        terminators = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.paraphrase_model, cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            token=args.hf_token,
            device_map=args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.paraphrase_model, cache_dir=args.cache_dir, token=args.hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    # also get the target tokenizer
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer, cache_dir=args.cache_dir, token=args.hf_token)
    target_tokenizer.pad_token = target_tokenizer.eos_token

    with open(args.input, 'rb') as file:
        canaries = pickle.load(file)
    
    results = []
    # generate paraphrases
    for canary in tqdm(canaries):
        # get the raw text
        seq = target_tokenizer.decode(canary)
        paraphrases = paraphrase_sequence(seq, model=model, tokenizer=tokenizer, terminators=terminators)
        paraphrases_tokenized = []
        for paraphrase in paraphrases:
            # tokenize the paraphrase
            tokens = target_tokenizer.encode(paraphrase)
            paraphrases_tokenized.append(tokens)
            
        print(f"Original: {seq}")
        for i, paraphrase in enumerate(paraphrases):
            print(f"Paraphrase {i+1}: {paraphrase}")
        
        results.append({"original": canary, "variations": paraphrases_tokenized})

    with open(args.output, "wb") as file:
        pickle.dump(results, file)

    print(f"Finished. Generated {len(canaries)} samples")
