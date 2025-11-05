
import pickle
import numpy as np
from sklearn.metrics import roc_curve
import torch
from typing import Sequence
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from utils import compute_perplexity, ratio_auc, min_k_prob
import zlib
import argparse

LLAMA_TOKENIZER_PATH = "SOME_DATA_DIR/Llama-2-7b-hf/"
LLAMA_MODEL_PATH = "SOME_DATA_DIR/Llama-2-7b-hf/"

def ppl_from_text(texts, model, tokenizer, device):
    tokens = tokenizer.batch_encode_plus(texts, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
    ppl = compute_perplexity(
        model,
        tokens.input_ids,
        tokens.attention_mask,
        ignore_prefix=None
    )

    return ppl

def tpr_at_fpr(members, non_members, target_fpr):
    y = list(members) + list(non_members)
    y_true = [0] * len(non_members) + [1] * len(members)
    fpr, tpr, _ = roc_curve(y_true, y)

    index = np.abs(fpr - target_fpr).argmin()
    return tpr[index]

def main(args):
    
    # load the models
    llama_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_TOKENIZER_PATH)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_model = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_PATH).to(args.ref_device)
    
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)
    target_tokenizer.pad_token = target_tokenizer.eos_token
    target_tokenizer.padding_side = "right"
    
    # load the members and non-members
    with open(args.path_to_members, 'rb') as f:
        og_canaries = pickle.load(f)
        
    og_canary_texts = [target_tokenizer.decode(og_canary) for og_canary in og_canaries]
    og_canary_texts_lower = [x.lower() for x in og_canary_texts]

    # and the non member text
    with open(args.path_to_non_members, 'rb') as f:
        non_members = pickle.load(f)

    non_member_texts = [target_tokenizer.decode(non_member) for non_member in non_members]
    non_member_texts_lower = [x.lower() for x in non_member_texts]
    
    all_aucs = defaultdict(dict)
    if 'nrep_' in args.variable:
        start = int(args.variable.split('_')[1])
        end = int(args.variable.split('_')[2])
        range_var = range(start, end+1)
    elif args.variable == 'nrep':
        range_var = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    elif args.variable == 'R':
        range_var = (1, 5, 10, 15, 20, 25, 50, 75)
    elif args.variable == 'n':
        range_var = (1, 2, 5, 10, 20, 50)
    elif 'tau' in args.variable: # expect 'N_tau'
        n = int(args.variable.split('_')[0])
        if n == 10:
            range_var = (2, 3, 4, 5, 6, 7, 8)
        else:
            range_var = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        
    for var in range_var:

        print(f"{args.variable} = {var}")

        score_members = {}
        score_non_members = {}

        target_path = args.generic_target_model.replace("XX", str(var))
        target_model = AutoModelForCausalLM.from_pretrained(target_path).to(args.target_device)

        # first: non members
        print("Running on non members...")
        non_member_target_ppl = ppl_from_text(non_member_texts, target_model, target_tokenizer, args.target_device)
        non_member_llama_ppl = ppl_from_text(non_member_texts, llama_model, llama_tokenizer, args.ref_device)
        non_member_lower_target_ppl = ppl_from_text(non_member_texts_lower, target_model, target_tokenizer, args.target_device)

        target_tokens_non_members = target_tokenizer.batch_encode_plus(non_member_texts, return_tensors="pt", 
                                                                       padding="longest", add_special_tokens=False).to(args.target_device)
        non_member_zlib_entropy = [len(zlib.compress(x.encode()))/len(x) for x in non_member_texts]
            
        score_non_members["ratio"] = non_member_target_ppl / non_member_llama_ppl
        score_non_members["loss"] = non_member_target_ppl
        score_non_members["lowercase"] = non_member_target_ppl / non_member_lower_target_ppl
        score_non_members["minkprob"] = -min_k_prob(target_model, target_tokens_non_members.input_ids, target_tokens_non_members.attention_mask)
        score_non_members["zlib"] = np.log(non_member_target_ppl) / non_member_zlib_entropy

        # now: non members
        print("Running on canaries...")
        og_canary_target_ppl = ppl_from_text(og_canary_texts, target_model, target_tokenizer, args.target_device)
        og_canary_llama_ppl = ppl_from_text(og_canary_texts, llama_model, llama_tokenizer, args.ref_device)
        og_canary_lower_target_ppl = ppl_from_text(og_canary_texts_lower, target_model, target_tokenizer, args.target_device)

        target_tokens_og_canary = target_tokenizer.batch_encode_plus(og_canary_texts, return_tensors="pt", 
                                                                     padding="longest", add_special_tokens=False).to(args.target_device)
        og_canary_zlib_entropy = [len(zlib.compress(x.encode()))/len(x) for x in og_canary_texts]

        score_members["ratio"] = og_canary_target_ppl / og_canary_llama_ppl
        score_members["loss"] = og_canary_target_ppl
        score_members["lowercase"] = og_canary_target_ppl / og_canary_lower_target_ppl
        score_members["minkprob"] = -min_k_prob(target_model, target_tokens_og_canary.input_ids, target_tokens_og_canary.attention_mask)
        score_members["zlib"] = np.log(og_canary_target_ppl) / og_canary_zlib_entropy

        for mia in score_members:
            auc = ratio_auc(members=score_members[mia], non_members=score_non_members[mia])
            tpr_at = tpr_at_fpr(score_members[mia], score_non_members[mia], target_fpr=0.1)
            print(f"{args.variable}={var}, MIA={mia}, AUC = {auc:.2f} - TPR @ 0.1 FPR = {tpr_at:.2f}")
            all_aucs[var][mia] = auc
            print("-------------------")
            
    # save the results
    with open(args.output, "wb") as f:
        pickle.dump(all_aucs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_device", type=str, default="cuda:0")
    parser.add_argument("--target_device", type=str, default="cuda:1")
    parser.add_argument("--generic_target_model", type=str, required=True)
    parser.add_argument("--target_tokenizer", type=str, required=True)
    parser.add_argument("--path_to_members", type=str, required=True)
    parser.add_argument("--path_to_non_members", type=str, required=True)
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)