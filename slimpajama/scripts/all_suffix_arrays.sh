set -uoe pipefail
ulimit -Sn 1000000

ROOT="/working/dir"
DATASET="slimpajama"
PARTS="20"

mkdir -p "logs"
mkdir -p "merge_scripts"

for i in {0..19}
do
  echo "Processing ${i}"
  mkdir -p "${ROOT}/tmps/tmp${i}"
  
  python py_src/dedupe/make_suffix_array.py \
  --input-path "${ROOT}/tokenized/${DATASET}_${i}_of_${PARTS}.train" \
  --tmp-path "${ROOT}/tmps/tmp${i}" \
  --total-jobs-mult 8 &> "logs/suffix_array_${i}_of_${PARTS}.log"

  tail -n 1 "logs/suffix_array_${i}_of_${PARTS}.log" > "merge_scripts/merge_${i}_of_${PARTS}.sh"

  bash "merge_scripts/merge_${i}_of_${PARTS}.sh" &> "logs/merge_${i}_of_${PARTS}.log"
    
  
  rm ${ROOT}/tokenized/${DATASET}_${i}_of_${PARTS}.train.part.*

  cat ${ROOT}/tmps/tmp${i}/out.table.bin.table.bin.00* > "${ROOT}/tmps/tmp${i}/out.table.bin"
  mv "${ROOT}/tmps/tmp${i}/out.table.bin" "${ROOT}/tokenized/${DATASET}_${i}_of_${PARTS}.train.table.bin"
  rm ${ROOT}/tmps/tmp${i}/out.table.bin.table.bin.00*
done
