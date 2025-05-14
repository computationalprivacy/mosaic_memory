# Scanning for near duplicates in SlimPajama

This repo provides the code for analyzing near-duplicates present in the SlimPajama dataset.
It's a multi-step process, which is best done on the machine with at least 80 CPUs and enough memory to fit a
significant portion of the dataset into memory.

This code is derived from https://github.com/google-research/deduplicate-text-datasets

## Step 1: Build the rust code

Install rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Build:
```bash
cargo build
```

## Step 2: Download and process SlimPajama

The entire dataset is ~900GB in size, and deduplication process requires multiple
times over the dataset size to fit into memory. We split the dataset into 20 subsets and process the deduplication
in each subset individually, and then merging hihgly repeated sequences.

First, you'd need to download SlimPajama. While it can be done as a part of the processing script, we find
pre-downloading the data to be more stable.

git clone the repo into the directory of your choosing (it will download ~900GB of data)

```bash
cd /path/to/slimpajama && git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```

Then, run the processing script, tokenizing the dataset and converting it into the format for deduplication:

```bash
python py_src/dedupe/load_slimpajama.py 
--slimpajama-path cd /path/to/slimpajama \
--save-dir /working/dir/tokenized \
--name slimpajama \
--tokenize \
--parts 20 \
--num-proc 80
```

It will take a while - first huggingface will cache the dataset for faster random access, 
and then launching the tokenization.

It should create 20 tokenized subsets in the `/working/dir/tokenized` dir: e.g. `slimpajama_0_of_20.train` (and slimpajama_0_of_20.train.size) with document sizes.

## Step 3: Find (exact) duplicates

We'll now build suffix arrays for deduplication. Update `/working/dir/` in `scripts/all_suffix_arrays.sh` 
and `scripts/all_self_similar.sh` with the actual dir containing tokenized slimpajama.

```bash
bash scripts/all_suffix_arrays.sh

bash scripts/all_self_similar.sh
```

This step is the longest - roughly 3-4 hours per subset (for 20 subsets), and the most resource intensive.

As a result, you'll get a number of binary files in `/working/dir/caches/cache*` containing positions and counts
of all exact duplicates in each subset.

## Step 4: Prepare for near-duplicate scan.

Scanning for near-duplicates, directly computing Hamming/Levenshtein distances is extremely expensive and cannot be
done on the full dataset scale. We make several assumptions, allowing us to reduce the computational cost to a
feasible level.

We select a small group of "target sequences", and we'll be scanning the dataset only looking for near duplicates
of these sequences. We sample target sequences based on the number of times they are duplicated exactly in the dataset.

First, we sample a number of exact duplicates present in one of the chunks (`chunk0`) and for each find the number of times 
they're duplicates in the entire dataset, across all chunks.

```bash
mkdir -p /working/dir/queries/

# --target-counts 5 50 500 - roughly corresponding to 100, 1000, 10_000 final buckets (20x chunks)
# --length 100 - in tokens
# --n-per-bucket 10000 - this step is relatively cheap, it's better to overshoot
# --bytes-per-record 5 - depending on the dataset size, byte-packing allocated different number of byter per one index
# for slimpajama it's 5

python py_src/near_duplicates/build_query.py \ 
--output-dir /working/dir/queries/ \
--dups-dir /working/dir/caches/cache0/ \
--ds-path /working/dir/tokenized/slimpajama_0_of_20.train \
--target-counts 5 50 500 \
--length 100 \
--n-per-bucket 10000 \
--bytes-per-record 5
```

This creates two files in `/working/dir/queries/`: sorted list of `positions.pkl`, and a log file from calling `count-occurences-multi`, where one line corresponds to one input position.

We now sample the required number of target sequences, given a full dataset duplication counts.

```bash
python py_src/near_duplicates/build_targets.py \
--output-path /working/dir/target_sequences.pkl \
--positions-path /working/dir/queries/positions.pkl \
--counts-dir /working/dir/queries/counts/ \
--ds-path /working/dir/tokenized/slimpajama_0_of_20.train \
--target-buckets 100 1000 10000 \
--target-bucket-tolerance 0.01 \
--length 100 \
--n-per-bucket 100 \
--uniq-token-min 50
```

Finally, we launch the scan. Again, for computational reasons we only scan one chunk (5% of the dataset) and 
extrapolate the results

```bash
python py_src/near_duplicates/scan.py \
--ds-path /working/dir/tokenized/slimpajama_0_of_20.train \
--save-dir /working/dir/scan/ \
--target-sequences-path /working/dir/target_sequences.pkl \
```

This produces a number of files in `/working/dir/scan/` - all that is needed to build the final graphs 
using `plot_near_duplicates.py`

