set -uoe pipefail

ROOT="/working/dir"
DATASET="slimpajama"
PARTS="20"

for i in {0..19}
do
  echo "Processing ${i}"

  mkdir -p "${ROOT}/caches/cache${i}"
  
  cargo run self-similar \
  --data-file "${ROOT}/tokenized/${DATASET}_${i}_of_${PARTS}.train" \
  --length-threshold 200 \
  --cache-dir "${ROOT}/caches/cache${i}" \
  --num-threads 80
done

