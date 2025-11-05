#!/bin/bash

for T in 1 5 10 15 20 25 50 75
do
    python src/gen_variations.py \
    --tokenizer "EleutherAI/gpt-neo-1.3B" \
    --input "SOME_DATA_DIR/members.pickle" \
    --output "SOME_DATA_DIR/near_dupls_members_diff_indices_topkrandom_T=$T.pickle" \
    --num-variations 9 \
    --num-injection-points $T \
    --no-replace-same-indices \
    --candidate-gen-strategy "mlm" \
    --topk=10 \
    --seed=42 \
    --device="cuda:0"
done