from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import logging
import numpy as np
from collections import Counter


@dataclass
class Sequence:
    pos: int
    tokens: list[int]
    n_rep: int
    bucket: int
    candidates: List[Tuple[int, int]] = field(default_factory=list)
    near_duplicates: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))



def bytes_to_ints(byte_data, bytes_per_int, byteorder="little"):
    ints = []
    for i in range(0, len(byte_data), bytes_per_int):
        chunk = byte_data[i : i + bytes_per_int]
        ints.append(int.from_bytes(chunk, byteorder=byteorder))
    return ints


def load_dataset(path):
    size_path = path + ".size"

    with open(size_path, "rb") as f:
        ds_size = f.read()
        ds_size = bytes_to_ints(ds_size, 8)

    with open(path, "rb") as f:
        ds = f.read()

    logging.info(f"Loaded total {len(ds)} bytes ({len(ds_size)} documents)")

    return ds, ds_size


def hamming_distance(arr1: List[int], arr2: List[int]) -> int:
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have equal length")
        
    return sum(1 for x, y in zip(arr1, arr2) if x != y)


def edit_distance(seq1: Sequence, seq2: Sequence) -> int:
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
        
    prev_row = list(range(len(seq1) + 1))
    curr_row = [0] * (len(seq1) + 1)
    
    for j in range(1, len(seq2) + 1):
        curr_row[0] = j
        
        for i in range(1, len(seq1) + 1):
            if seq1[i-1] == seq2[j-1]:
                curr_row[i] = prev_row[i-1]
            else:
                curr_row[i] = min(prev_row[i-1] + 1,  # substitution
                                prev_row[i] + 1,     # deletion
                                curr_row[i-1] + 1)   # insertion
        
        prev_row, curr_row = curr_row, prev_row
        
    return prev_row[-1]

METRICS = {
    "hamming": hamming_distance,
    "edit": edit_distance,
}

def log_target_sequnces_stats(target_sequences):
    # Log statistics about target sequences
    bucket_stats = {}
    for seq in target_sequences:
        bucket = seq.bucket if seq.bucket != -1 else 'random'
        if bucket not in bucket_stats:
            bucket_stats[bucket] = {'count': 0, 'n_rep_sum': 0, 'uniq_tokens': []}
        bucket_stats[bucket]['count'] += 1
        bucket_stats[bucket]['n_rep_sum'] += seq.n_rep if seq.n_rep != -1 else 0
        bucket_stats[bucket]['uniq_tokens'].append(len(set(seq.tokens)))
    
    logging.info(f"Total {len(target_sequences)} target sequences")

    for bucket, stats in bucket_stats.items():
        avg_n_rep = stats['n_rep_sum'] / stats['count'] if bucket != 'random' else -1
        uniq_tokens = stats['uniq_tokens']
        logging.info(f"Bucket {bucket}: count={stats['count']}, avg_n_rep={avg_n_rep:.1f}, "
                    f"unique tokens: avg={np.mean(uniq_tokens):.1f}, min={min(uniq_tokens)}, max={max(uniq_tokens)}")

    # Log example tokens from first sequence
    logging.info(f"Example tokens from first sequence: {target_sequences[0].tokens[:10]}")


def merge_results(acc, new_results):
    if len(acc) != len(new_results):
        raise ValueError()
    
    for acc_seq, new_seq in zip(acc, new_results):
        if acc_seq.pos != new_seq.pos:
            raise ValueError()

        for metric in METRICS:
            acc_seq.near_duplicates[metric].extend(new_seq.near_duplicates.get(metric, []))
    
    return acc



def print_near_duplicate_summary(seqs, pre_bucket = None):
    if pre_bucket is not None:
        seqs = [seq for seq in seqs if seq.bucket == pre_bucket]
    
    for metric in ("hamming", "edit"):
        summary = Counter()
        for seq in seqs:
            for _, distance in seq.near_duplicates.get(metric,[]):
                bucket = (distance // 10)*10
                summary[bucket] += 1
        
        summary_strs = []
        keys = sorted(summary.keys())
        for i in range(len(keys)):
            summary_strs.append(f"{keys[i]}-{keys[i]+9}: {summary[keys[i]]}")
        
        summary = " | ".join(summary_strs)
        if pre_bucket is not None:
            metric = f"({pre_bucket}) {metric}"
        
        print(f"{metric} => {summary}")