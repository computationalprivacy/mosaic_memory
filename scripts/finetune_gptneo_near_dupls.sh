#!/bin/bash

for R in 1 5 10 15 20 25 50 75
do
CUDA_VISIBLE_DEVICES=0 python src/fine_tune_model.py --model "EleutherAI/gpt-neo-1.3B" --learning_rate_e6 20 --device="cuda:0" \
   --training_data="SOME_DATA_DIR/books_w_neardupl_canaries_journal_diff_indices_topkrandom_T${R}_100" \
--path_to_members="SOME_DATA_DIR/members.pickle" \
--path_to_non_members="SOME_DATA_DIR/non_members.pickle" \
--batch-size=2 --accumulate-steps=1 --run_name="gptneo1B_near_duplicates_diff_indices_topkrandom_R${R}_lr2e5" 
done

