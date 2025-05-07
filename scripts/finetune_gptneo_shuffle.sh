#!/bin/bash

for tau in 1 2 3 4 5 6 7 8 9
do
CUDA_VISIBLE_DEVICES=0 python src/fine_tune_model.py --model "EleutherAI/gpt-neo-1.3B" --learning_rate_e6 20 --device="cuda:0" \
   --training_data="SOME_DATA_DIR/kendall_dist_0${tau}_ngram_2" \
--path_to_members="SOME_DATA_DIR/members.pickle" \
--path_to_non_members="SOME_DATA_DIR/non_members.pickle" \
--batch-size=2 --accumulate-steps=1 --run_name="gptneo1B_near_duplicates_kendall_dist_0${tau}_ngram_2_lr2e5" 
done
