# set -uoe pipefail
ulimit -Sn 1000000

ROOT="/root/mosaic_memory/slimpajama"
TOKENIZED_ROOT="/workspace/igors/mosaic_memory/data/"
QUERY_PATH="${ROOT}/queries/query"

DATASET="slimpajama"
PARTS="20"

mkdir -p "logs/counts"
mkdir -p "merge_scripts"
mkdir -p "results"

for i in {1..19}
do
  echo "Processing ${i}"
  mkdir -p "${ROOT}/tmps/tmp${i}"
  
  python py_src/dedupe/make_suffix_array.py \
  --input-path "${TOKENIZED_ROOT}/${DATASET}_${i}_of_${PARTS}.train" \
  --tmp-path "${ROOT}/tmps/tmp${i}" \
  --num-threads 32 \
  --total-jobs-mult 8 &> "logs/suffix_array_${i}_of_${PARTS}.log"

  tail -n 1 "logs/suffix_array_${i}_of_${PARTS}.log" > "merge_scripts/merge_${i}_of_${PARTS}.sh"

  bash "merge_scripts/merge_${i}_of_${PARTS}.sh" &> "logs/merge_${i}_of_${PARTS}.log"
    
  
  rm ${TOKENIZED_ROOT}/${DATASET}_${i}_of_${PARTS}.train.part.*

  cat ${ROOT}/tmps/tmp${i}/out.table.bin.table.bin.00* > "${ROOT}/tmps/tmp${i}/out.table.bin"
  mv "${ROOT}/tmps/tmp${i}/out.table.bin" "results/${DATASET}_${i}_of_${PARTS}.train.table.bin"
  rm ${ROOT}/tmps/tmp${i}/out.table.bin.table.bin.00*

  cp "${TOKENIZED_ROOT}/${DATASET}_${i}_of_${PARTS}.train" "results/"
  cp "${TOKENIZED_ROOT}/${DATASET}_${i}_of_${PARTS}.train.size" "results/"

  ./target/debug/dedup_dataset count-occurrences-multi \
    --data-file "results/${DATASET}_${i}_of_${PARTS}.train" \
    --query-file "${QUERY_PATH}" \
    --load-disk &> "logs/counts/${i}_of_20.cnt"

  rm "results/${DATASET}_${i}_of_${PARTS}.train"
  rm "results/${DATASET}_${i}_of_${PARTS}.train.size"
  rm "results/${DATASET}_${i}_of_${PARTS}.train.table.bin"
done
