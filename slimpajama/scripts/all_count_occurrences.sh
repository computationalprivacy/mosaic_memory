set -uoe pipefail

ROOT="/data/igor/deduplicate-text-datasets"
DATA_ROOT="/home/igor/rds/ephemeral/datasets"

for i in {0..19}
do
    ./target/debug/dedup_dataset count-occurrences-multi \
    --data-file "${DATA_ROOT}/tokenized/slimpajama_${i}_of_20.train" \
    --query-file "${ROOT}/queries/all_buckets" &> "${ROOT}/logs/counts/${i}_of_20.cnt"
done