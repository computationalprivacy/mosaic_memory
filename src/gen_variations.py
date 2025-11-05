import torch
import pickle
import numpy as np
from transformers import  AutoTokenizer, AutoModelForCausalLM, RobertaForMaskedLM
from tqdm import tqdm
import random
import argparse
import json
from copy import deepcopy

def sample_injection_indices(seq, num_injection_points, is_uniform_spread):
    if is_uniform_spread:
        segments = np.array_split(range(len(seq)), num_injection_points)
        return [random.choice(segment) for segment in segments]
    else:
        return random.sample(range(len(seq)), k=num_injection_points)

def build_single_position_candidates_mlm(tokens, topk, tokenizer, mlm_tokenizer, mlm_model, device):
    
    masked_text = []

    for i in range(len(tokens)):
        tmp = list(tokens)
        prefix_text = tokenizer.decode(tmp[:i]) 
        suffix_text = tokenizer.decode(tmp[i+1:])
        
        # now let's get the prediction for a token in between
        tmp_masked = prefix_text + mlm_tokenizer.mask_token + suffix_text
        masked_text.append(tmp_masked)
    
    mlm_inputs = mlm_tokenizer.batch_encode_plus(masked_text, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    mlm_inputs = {key: value.to(device) for key, value in mlm_inputs.items()}
    
    with torch.no_grad():
        logits = mlm_model(**mlm_inputs).logits
    # logits.shape = (100, 10X, vocab_size)
        
    # get the positions of the mask tokens
    mask_token_indices = (mlm_inputs['input_ids'] == mlm_tokenizer.mask_token_id).nonzero(as_tuple=False).cpu().numpy()
    # this is of shape (num_examples, 2) where each row is (example_index, position_in_example)
    
    candidates = {}
    # let's now run through the selected mlm tokens to replace each target token
    for i in range(len(tokens)):
        
        # get the position of the mask token for this example
        assert mask_token_indices[i][0] == i
        mask_token_pos = mask_token_indices[i][1]
                
        top_indices = logits[i, mask_token_pos].topk(topk).indices.cpu().numpy() 
        top_text = [mlm_tokenizer.decode([idx]) for idx in top_indices]
        
        # now let's select the replacement token, making sure it's not the same as the original one
        original_token = tokenizer.decode(tokens[i])
        replacement_tokens = [tokenizer.encode(text, add_special_tokens=False) for text in top_text if text != original_token]
                
        candidates[i] = (replacement_tokens, None)

    return candidates

def build_single_position_candidates_mlm_random(tokens, tokenizer, mlm_tokenizer, mlm_vocab_size):
    ret = {}
    for i in range(len(tokens)):
        sampled_mlm_tokens = random.sample(range(mlm_vocab_size), k=50)
        sampled_text = [mlm_tokenizer.decode([sampled_mlm_token]) for sampled_mlm_token in sampled_mlm_tokens]
        replacement_tokens = [tokenizer.encode(text, add_special_tokens=False) for text in sampled_text]
        ret[i] = (replacement_tokens, None)
    return ret

def build_single_position_candidates_random(tokens, vocab_size, topk=50):
    ret = {
        i: (random.sample(range(vocab_size), k=topk), None) for i in range(len(tokens))
    }
    return ret


def build_candidates(tokens, tokenizer, mlm_tokenizer, mlm_model, device, args):
    if args.candidate_gen_strategy == "mlm":
        return build_single_position_candidates_mlm(
            tokens=tokens, 
            topk=args.topk,
            tokenizer=tokenizer,
            mlm_tokenizer=mlm_tokenizer,
            mlm_model=mlm_model,
            device=device,
        )
    elif args.candidate_gen_strategy == "mlm_random":
        return build_single_position_candidates_mlm_random(
            tokens=tokens,
            tokenizer = tokenizer,
            mlm_tokenizer=mlm_tokenizer,
            mlm_vocab_size=mlm_tokenizer.vocab_size,
        )
    elif args.candidate_gen_strategy == "random":
        return build_single_position_candidates_random(
            tokens=tokens,
            vocab_size=tokenizer.vocab_size,
        )
    else:
        raise ValueError()


def process_sequence_basic(tokens, tokenizer, mlm_tokenizer, mlm_model, device, args):
    
    candidates = build_candidates(
        tokens=tokens,
        tokenizer=tokenizer,
        mlm_tokenizer=mlm_tokenizer, 
        mlm_model=mlm_model,
        device=device,
        args=args,
    )

    variations = []
    injection_indices = None

    for _ in range(args.num_variations):
        if injection_indices is None or (not args.replace_same_indices):
            injection_indices = sample_injection_indices(tokens, args.num_injection_points, args.enable_uniform_spread)

        new_tokens = deepcopy(tokens)
        correction_idx = 0
        for idx in injection_indices:
            indices, probs = candidates[idx]
            candidate_idx = np.random.choice(range(len(indices)), p=probs)
            candidate = indices[candidate_idx]
            
            # candidate can now be a list of tokens
            new_tokens = new_tokens[:idx+correction_idx] + candidate + new_tokens[idx+1+correction_idx:]
            
            if len(candidate) > 1:
                correction_idx += len(candidate) - 1
        assert len(new_tokens) == len(tokens) + correction_idx
        variations.append(new_tokens)

    return {
        "original": tokens,
        "variations": [{"tokens": x} for x in variations]
    }

def main(args):
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta_tokenizer.pad_token = roberta_tokenizer.eos_token
    roberta_model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.input, 'rb') as file:
        canaries = pickle.load(file)

    result = []
    for seq_tokens in tqdm(canaries):

        r = process_sequence_basic(
                tokens=seq_tokens,
                tokenizer=tokenizer,
                mlm_tokenizer=roberta_tokenizer, 
                mlm_model=roberta_model,
                device=device,
                args=args,
        )
        if r is not None:
            result.append(r)

    if args.output_format == "pickle":
        pickle.dump(result, open(args.output, 'wb'))
    elif args.output_format == "json":
        json.dump(result, open(args.output, 'w'), indent=4)
    else:
        raise ValueError()

    print(f"Finished. Generated {sum([len(x['variations']) for x in result])} variations for {len(result)} sequences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer to use, assuming the same tokenizer is used for both the model and the input data")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input file path (format as produced by gen_canaries.py)")
    parser.add_argument("--num-variations", type=int, required=True, help="Number of near duplicates to be generated")
    parser.add_argument("--num-injection-points", type=int, required=True,
                        help="Number of units (words or tokens) that will be switched from the original sequence")
    parser.add_argument('--replace-same-indices', action=argparse.BooleanOptionalAction, required=True,
                        help="If true, all near duplicates generated for a given sequence will differ in the same"
                        "positions. Otherwise injection indices are sampled for each new near-duplicate")

    parser.add_argument("--topk", type=int, required=False, help="topk for individual word/token random sampling"
                        "(only used if candidates are generated with mlm). "
                        "If set to 0, candidate will be sampled from the full distribution")

    parser.add_argument("--enable-uniform-spread", action="store_true",
                        help="If enabled, injection points will be roughly uniformly spread across the sequence."
                        "Otherwise injection indices are picked at random")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-format", type=str, choices=["pickle", "json"], default="pickle")
    parser.add_argument('--candidate-gen-strategy', type=str, choices=[
                        "random", "mlm", "mlm_random"], default="mlm", help="Strategy to generate candidate words")

    args = parser.parse_args()
    
    if args.candidate_gen_strategy == "mlm" and args.topk is None:
        raise ValueError("--topk is required when using LM to generate candidates")

    main(args)
