#!/bin/bash

python src/gen_canaries.py --path-to-model="SOME_DATA_DIR/Llama-2-7b-hf/" \
                           --path-to-tokenizer="SOME_DATA_DIR/Llama-2-7b-hf/" \
                           --path-to-target-tokenizer="EleutherAI/gpt-neo-1.3B" \
                           --output="SOME_DATA_DIR/non_members.pickle" \
                           --seq-len=100 --num-canaries=100 --temp=1 \
                           --batch-size=50 --device="cuda:0" --seed=420


