#!/bin/bash

for ndup in 10 9 8 7 6 5 4 3 2 1
do
CUDA_VISIBLE_DEVICES=0 python src/fine_tune_model.py --model "EleutherAI/gpt-neo-1.3B" --learning_rate_e6 20 --device="cuda:0" \
   --training_data="SOME_DATA_DIR/books_w_exactdupl_canaries_decoder_nrep${ndup}" \
--path_to_members="SOME_DATA_DIR/members.pickle" \
--path_to_non_members="SOME_DATA_DIR/non_members.pickle" \
--batch-size=2 --accumulate-steps=1 --run_name="gptneo1B_exact_duplicates_nrep${ndup}_lr2e5" 
done

