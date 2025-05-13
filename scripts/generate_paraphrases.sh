#!/bin/bash

python src/paraphrase.py \
    --paraphrase-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --target-tokenizer "EleutherAI/gpt-neo-1.3B" \
    --input "SOME_DATA_DIR/members.pickle" \
    --output "SOME_DATA_DIR/paraphrases_members_mistral7B_instruct_v02.pickle" \
    --seed=42 \
    --device="cuda:0"