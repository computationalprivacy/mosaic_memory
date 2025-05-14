import os
import logging
import argparse
import glob
import time
from utils import bytes_to_ints, load_dataset
from collections import defaultdict
import numpy as np
from transformers import GPT2Tokenizer
import pickle

logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Starting build_query with args: {args}")
    start_time = time.time()
    
    dups_paths = sorted(glob.glob(args.dups_dir + "/dups_*"))
    sizes_paths = sorted(glob.glob(args.dups_dir + "/sizes_*"))
    logger.info(f"Found {len(dups_paths)} duplicate files and {len(sizes_paths)} size files")

    samples_per_chunk = args.n_per_bucket // len(dups_paths)
    if samples_per_chunk == 0:
        samples_per_chunk = 1
    
    all_buckets = defaultdict(list)
    logger.info(f"Target counts: {args.target_counts}, samples per chunk: {samples_per_chunk}")
    for i, (dups_path, sizes_path) in enumerate(zip(dups_paths, sizes_paths)):
        chunk_start_time = time.time()
        logger.info(f"Processing chunk {i+1}/{len(dups_paths)}: {dups_path}")
        with open(sizes_path, "rb") as f:
            dups_sizes = f.read()
            dups_sizes = bytes_to_ints(dups_sizes, args.bytes_per_record)

        with open(dups_path, "rb") as f:
            dups = f.read()
            dups = bytes_to_ints(dups, args.bytes_per_record)
        
        buckets = defaultdict(list)
        total_added = 0
        logger.info(f"Processing {len(dups)} positions from {dups_path}")
        for pos, count in zip(dups, dups_sizes):
            if count not in args.target_counts:
                continue

            if len(buckets[count]) >= samples_per_chunk:
                continue

            buckets[count].append(pos)
            total_added += 1

            if total_added >= samples_per_chunk * len(args.target_counts):
                break

        for k,v in buckets.items():
            all_buckets[k].extend(v)
        
        logger.info(f"Chunk {i+1} processed in {time.time() - chunk_start_time:.2f}s. Added {total_added} positions.")

    logger.info(f"Collected {sum(len(v) for v in all_buckets.values())} positions across {len(all_buckets)} buckets")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    logger.info(f"Loading dataset from {args.ds_path}")
    ds, _ = load_dataset(args.ds_path)
    final_positions = []

    query_path = os.path.join(args.output_dir, "query")
    logger.info(f"Writing query data to {query_path}")
    processed_count = 0
    with open(query_path, "wb") as f:
        for count, positions in all_buckets.items():
            logger.info(f"Processing bucket with count {count}, {len(positions)} positions")
            for pos in positions:
                if pos%2 == 1:
                    pos += 1
                
                text = tokenizer.decode(bytes_to_ints(ds[pos:pos+args.length * 2],2))
                arr = np.array(tokenizer.encode(text), dtype=np.uint16).view(np.uint8).tobytes()
                arr_length = np.array([len(arr)], dtype=np.uint32).view(np.uint32).tobytes()

                f.write(arr_length)
                f.write(arr)

                final_positions.append(pos)
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count} positions so far")

    positions_path = os.path.join(args.output_dir, "positions.pkl")
    logger.info(f"Writing {len(final_positions)} final positions to {positions_path}")
    with open(positions_path, "wb") as f:
        pickle.dump(final_positions, f)
    
    total_time = time.time() - start_time
    logger.info(f"build_query completed in {total_time:.2f}s")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dups-dir", type=str, required=True)
    parser.add_argument("--ds-path", type=str, required=True)
    parser.add_argument("--target-counts", type=int, nargs='+', default=None)
    parser.add_argument("--length", type=int, default=100, help="Target sequence length in tokens")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-per-bucket", type=int, default=10000, help="Number of target sequences per bucket")
    parser.add_argument("--bytes-per-record", type=int, default=5)

    args = parser.parse_args()
    main(args)