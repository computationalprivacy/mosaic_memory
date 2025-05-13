import argparse
import pickle
import random
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
        "You are an assistant tasked with rephrasing the provided text in 9 different ways. "
        "Keep the original meaning intact (including the original natural or code language), but "
        "rephrase each version as if you are replacing the sentence entirely. "
        "Number the rephrased sequences using 1. to 9. and separate each by '---', like this: "
        "'1. rephrase 1 --- 2. rephrase 2 --- .. --- 9. rephrase 9'.")

def paraphrase_sequence(seq, model, tokenizer, terminators):

    user_prompt = f"Can you rephrase the following sequence? '{seq}'"

    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
    ]
        
    succeeded = False
    while not succeeded:

        input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1]:]
        decoded = tokenizer.decode(response, skip_special_tokens=True)
        
        # remove everything before 1.
        decoded = re.sub(r'^[^1]*1\.', '1.', decoded)
        rephrases = [s.strip() for s in re.split(r'\s*---\s*', decoded) if s.strip()]
        rephrases = [re.sub(r'^\d+\.\s*', '', r) for r in rephrases]  # Remove leading numbers

        if len(rephrases) == 9:
            succeeded = True
        else:
            continue

    return rephrases

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--paraphrase-model", type=str, required=True)
    parser.add_argument("--target-tokenizer", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="SOME_DATA_DIR/cache/huggingface")
    parser.add_argument("--hf-token", type=str, default="SOME_HUGGINGFACE_TOKEN")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # load the paraphrase model
    model = AutoModelForCausalLM.from_pretrained(
        args.paraphrase_model, cache_dir=args.cache_dir,
        torch_dtype=torch.float32,
        token=args.hf_token,
        device_map=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.paraphrase_model, cache_dir=args.cache_dir, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    # also get the target tokenizer
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer, cache_dir=args.cache_dir, token=args.hf_token)
    target_tokenizer.pad_token = target_tokenizer.eos_token

    with open(args.input, 'rb') as file:
        canaries = pickle.load(file)
    
    # EOS tokens
    terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
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
        results.append({"original": canary, "variations": paraphrases_tokenized})

    with open(args.output, "wb") as file:
        pickle.dump(results, file)

    print(f"Finished. Generated {len(canaries)} samples")
