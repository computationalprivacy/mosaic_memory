#!/bin/bash

for SEED in 1 2 3 4 5
do
     python src/compute_mia_auc.py  \
     --ref_device="cuda:0" \
     --target_device="cuda:1" \
     --generic_target_model="SOME_DATA_DIR/EleutherAI_gpt-neo-1.3B_gptneo1.3B_exact_duplicates_seed${SEED}_nrepXX_lr2e5" \
     --target_tokenizer="EleutherAI/gpt-neo-1.3B" \
     --path_to_members="SOME_DATA_DIR/members_seed${SEED}.pickle" \
     --path_to_non_members="SOME_DATA_DIR/non_members_seed${SEED}0.pickle" \
     --variable="nrep" \
     --output="SOME_DATA_DIR/nrep_aucs_gptneo1B_exactduplicates_seed${SEED}_lr2e5.pickle"
done
