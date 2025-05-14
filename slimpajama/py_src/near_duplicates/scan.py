import argparse
import copy
import logging
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import (ProcessPoolExecutor,
                                as_completed)
from multiprocessing import shared_memory

import numpy as np
from tqdm import tqdm
from utils import (bytes_to_ints, load_dataset, log_target_sequnces_stats, METRICS, merge_results, print_near_duplicate_summary)

class CharCounter:

    def __init__(self, sequence):
        self.counter = Counter(sequence.tokens)
        self.sequence = sequence
        self.n_matches = 0
    
    def new_char_add(self, c):
        if c in self.counter:
            self.counter[c] -= 1

            if self.counter[c] >= 0:
                self.n_matches += 1

    def old_char_remove(self, c):
        if c in self.counter:
            self.counter[c] += 1

            if self.counter[c] > 0:
                self.n_matches -= 1

    def step(self, new_char, old_char):
        self.new_char_add(new_char)
        self.old_char_remove(old_char)
        return self.n_matches
    
    def __str__(self):
        return f"{str(self.counter)} | matched = {self.n_matches}"


def process_chunk(start_idx, end_idx, ds_size_segment, target_seqs, shm_name, max_distance, save_dir):
    target_seqs = copy.deepcopy(target_seqs)
    shm = shared_memory.SharedMemory(name=shm_name)

    try:

        for i in range(len(ds_size_segment)-1):
            start_pos = ds_size_segment[i]
            end_pos = ds_size_segment[i+1]
            doc = bytes_to_ints(shm.buf[start_pos:end_pos],2)

            counters = [CharCounter(seq) for seq in target_seqs]
            seq_len = len(target_seqs[0].tokens)

            if len(doc) < seq_len:
                continue
            
            for j in range(seq_len):
                new_char = doc[j]
                for counter in counters:
                    counter.new_char_add(new_char)
            
            for counter in counters:
                if counter.n_matches + max_distance >= seq_len:
                    candidate_pos = start_pos
                    candidate_dist = seq_len - counter.n_matches
                    counter.sequence.candidates.append((candidate_pos, candidate_dist))
                    

            for j in range(seq_len, len(doc)):
                new_char = doc[j]
                old_char = doc[j-seq_len]
                
                for counter in counters:
                    counter.step(new_char, old_char)

                    if counter.n_matches + max_distance >= seq_len:
                        candidate_pos = start_pos + (j+1-seq_len)*2
                        candidate_dist = seq_len - counter.n_matches
                        counter.sequence.candidates.append((candidate_pos, candidate_dist))

        for seq in target_seqs:
            seq.near_duplicates = process_candidates(seq.tokens, seq.candidates, shm, max_distance)
        
        filename = f"res_{start_idx}_{end_idx}.pkl"
        with open(os.path.join(save_dir, filename), 'wb') as f:
            pickle.dump(target_seqs, f)
        
    finally:
        shm.close()
    
    return target_seqs


def process_consecutive(tokens, candidates, shm, metric_fn, threshold, length=100):
    distances = []
    for candidate_pos in candidates:
        if candidate_pos % 2 != 0:
            raise ValueError("Candidate position must be even")
        
        candidate_tokens = bytes_to_ints(shm.buf[candidate_pos:candidate_pos+length*2],2)
        distance = metric_fn(tokens, candidate_tokens)

        if distance <= threshold:
            distances.append((candidate_pos, distance))
    
    res = []

    while len(distances) > 0:
        max_idx = np.argmax([-x[1] for x in distances])
        res.append(distances[max_idx])
        max_pos = distances[max_idx][0]
        
        distances = [x for x in distances if (x[0] + length*2 <= max_pos or x[0] >= max_pos + length*2)]
    
    return res

def process_candidates(tokens, candidates, shm, threshold, length=100):
    if len(candidates) == 0:
        return defaultdict(list)
    
    results = defaultdict(list)
    pos_candidates = [x[0] for x in candidates]
    curr = [pos_candidates[0]]
    
    for i in range(1, len(pos_candidates)):
        if pos_candidates[i] > curr[-1] + 2*length:
            for metric, metric_fn in METRICS.items():
                results[metric].extend(
                    process_consecutive(
                        tokens=tokens, 
                        candidates=curr, 
                        shm=shm,
                        metric_fn=metric_fn,
                        threshold=threshold,
                        length=length,
                     )
                )
            curr = [pos_candidates[i]]
        else:
            curr.append(pos_candidates[i])


    for metric, metric_fn in METRICS.items():
        results[metric].extend(
            process_consecutive(
                tokens=tokens, 
                candidates=curr, 
                shm=shm,
                metric_fn=metric_fn,
                threshold=threshold,
                length=length,
             )
        )

    return results


def make_chunks(ds_size, n_proc, chunks_per_proc):
    n_chunks = n_proc * chunks_per_proc
    chunk_size = len(ds_size) // n_chunks

    chunks = []

    for chunk_id in range(n_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size

        if chunk_id == n_chunks - 1:
            end_idx = len(ds_size) - 1

        chunks.append((start_idx, end_idx))
    
    return chunks


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="Scan dataset for near duplicates")
    parser.add_argument("--ds-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--target-sequences-path", type=str, required=True)
    parser.add_argument("--n-proc", type=int, default=80)
    parser.add_argument("--chunks-per-proc", type=int, default=1000)
    parser.add_argument("--print-freq", type=int, default=800)
    parser.add_argument("--start-from", type=int, default=None)
    parser.add_argument("--max-distance", type=int, default=50)
    parser.add_argument("--missing-only", action="store_true")
    args = parser.parse_args()

    ds, ds_size = load_dataset(args.ds_path)

    shm = shared_memory.SharedMemory(create=True, size=len(ds))

    with open(args.target_sequences_path, 'rb') as f:
        target_sequences = pickle.load(f)
    
    log_target_sequnces_stats(target_sequences)

    try:
        shm.buf[:len(ds)] = ds

        chunks = make_chunks(ds_size, args.n_proc, args.chunks_per_proc)
        if args.start_from is not None:
            chunks = [x for x in chunks if x[0] >= args.start_from]

        if args.missing_only:
            existing_files = set(os.listdir(args.save_dir))

            new_chunks = []
            for chunk in chunks:
                filename = f"res_{chunk[0]}_{chunk[1]}.pkl"
                if filename not in existing_files:
                    new_chunks.append(chunk)

            chunks = new_chunks

        logging.info(f"Processing {len(chunks)} chunks. Each {chunks[0][1]-chunks[0][0]} documents")

        results_acc = copy.deepcopy(target_sequences)
        buckets = sorted(set([x.bucket for x in target_sequences]))
        completed = 0

        with ProcessPoolExecutor(max_workers=args.n_proc) as pool:
            futures = [
                pool.submit(
                    process_chunk, 
                    start_idx, # start_idx
                    end_idx, # end_idx
                    ds_size[start_idx:end_idx], # ds_size_segment
                    target_sequences, # target_seqs
                    shm.name , # shm_name
                    args.max_distance, # max_distance,
                    args.save_dir, # save_dir,
                ) for start_idx, end_idx in chunks
            ]

            with tqdm(total=len(chunks)) as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results_acc = merge_results(results_acc, result)
                        completed += 1

                        if completed % args.print_freq == 0:
                            print(f"Step: {completed} / {len(chunks)}")
                            print(f"Total sequences: {len(results_acc)}")
                            print_near_duplicate_summary(results_acc)
                            for bucket in buckets:
                                print("--")
                                print(f"Bucket {bucket}")
                                print_near_duplicate_summary(results_acc, pre_bucket=bucket)
                            print("=====================================")

                    except Exception as e:
                        print(f'Task failed: {e}')
                        raise e
                    pbar.update(1)
    finally:
        shm.close()
        shm.unlink()

    