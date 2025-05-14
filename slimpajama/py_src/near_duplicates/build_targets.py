import argparse
import logging
import pickle
import os
import random
from utils import bytes_to_ints, load_dataset, Sequence, log_target_sequnces_stats
import numpy as np


def load_counts(dir_path):
    total_counts = None

    for filename in os.listdir(dir_path):
        if filename.startswith("."):
            continue

        path = os.path.join(dir_path, filename)

        with open(path, "r") as f:
            lines = f.readlines()
            if total_counts is None:
                total_counts = [0] * len(lines)

            if len(lines) != len(total_counts):
                raise ValueError(
                    f"Unexpected number of lines in {path}. Found: {len(lines)}. Expected: {len(total_counts)}")

            for j, line in enumerate(lines):
                count = int(line.split()[-1])
                total_counts[j] += count

    return total_counts


def sample_random_sequences(ds, ds_size, n, length, uniq_token_min=None):
    results = []
    while len(results) < n:
        doc_pos = random.choice(range(len(ds_size)))
        pos1, pos2 = ds_size[doc_pos], ds_size[doc_pos+1]

        # x2 because doc is in bytes and length is in tokens
        if pos2 - length*2 <= pos1:
            continue

        seq_pos = random.choice(range(pos1, pos2 - length*2))
        if seq_pos % 2 != 0:
            seq_pos += 1

        tokens = bytes_to_ints(ds[seq_pos:seq_pos+length*2], 2)
        if uniq_token_min is not None and len(set(tokens)) < uniq_token_min:
            continue

        sequence = Sequence(
            pos=seq_pos,
            tokens=tokens,
            n_rep=-1,
            bucket=-1
        )

        results.append(sequence)

    return results


def sample_bucket(ds, source_positions, source_counts, n, target_count, target_tolerance, length, uniq_token_min=None):
    if len(source_positions) != len(source_counts):
        raise ValueError(
            f"source_positions and source_counts must have the same length. "\
              "Found: {len(source_positions)} and {len(source_counts)}"
        )

    results = []
    count_lower = round(target_count * (1 - target_tolerance))
    count_upper = round(target_count * (1 + target_tolerance))

    count_indices = [(i, cnt) for i, cnt in enumerate(source_counts) if count_lower <= cnt <= count_upper]
    filtered_positions = [(source_positions[i],cnt) for i,cnt in count_indices]

    for pos,cnt in filtered_positions:
        if pos%2 != 0:
            pos += 1

        token_bytes = bytes_to_ints(ds[pos:pos+2*length],2)

        if uniq_token_min is not None and len(set(token_bytes)) < uniq_token_min:
            continue 

        sequence = Sequence(
            pos=pos,
            tokens=token_bytes,
            n_rep=cnt,
            bucket=target_count
        )

        results.append(sequence)

    results = random.sample(results, k=n)
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--positions-path", type=str, required=True)
    parser.add_argument("--counts-dir", type=str, required=True)
    parser.add_argument("--ds-path", type=str, required=True)
    parser.add_argument("--target-buckets", type=int, nargs='+', default=[10, 100, 1000, 10_000])
    parser.add_argument("--target-bucket-tolerance", type=float, default=0.01)
    parser.add_argument("--length", type=int, default=100, help="Target sequence length in tokens")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-per-bucket", type=int, default=100, help="Number of target sequences per bucket")
    parser.add_argument("--uniq-token-min", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.positions_path, 'rb') as f:
        positions = pickle.load(f)

    counts = load_counts(args.counts_dir)
    logging.info(f"Loaded total counts for {len(positions)} sequences. First: {counts[:5]}. Last: {counts[-5:]}")

    ds, ds_size = load_dataset(args.ds_path)

    target_sequences = sample_random_sequences(
        ds=ds,
        ds_size=ds_size,
        length=args.length,
        n=args.n_per_bucket,
        uniq_token_min=args.uniq_token_min,
    )

    for bucket in args.target_buckets:
        target_sequences += sample_bucket(
            ds=ds,
            source_positions=positions,
            source_counts=counts,
            n=args.n_per_bucket,
            target_count=bucket,
            target_tolerance=args.target_bucket_tolerance,
            length=args.length,
            uniq_token_min=args.uniq_token_min,
        )
    
    log_target_sequnces_stats(target_sequences)
    
    with open(args.output_path, 'wb') as f:
        pickle.dump(target_sequences, f)
        logging.info(f"Saved {len(target_sequences)} target sequences to {args.output_path}")
