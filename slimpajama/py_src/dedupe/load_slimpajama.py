import os
import struct
import numpy as np
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Load a dataset.")
parser.add_argument("--slimpajama-path", type=str, required=True)
parser.add_argument("--save-dir", type=str, required=True)
parser.add_argument("--parts", type=int, required=True)
parser.add_argument("--num-proc", type=int, default=80)
args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
PRE_STEP = b"\xff\xff"
POST_STEP = b""

N_CHUNKS = 10
N_FILES = {
    1: 5912,
    2: 5911,
    3: 5919,
    4: 5917,
    5: 5933,
    6: 5915,
    7: 5906,
    8: 5921,
    9: 5920,
    10: 5912
}
BASE_PATH=os.path.join(args.slimpajama_path, "train/chunk{chunk}/example_train_{i}.jsonl.zst")


if args.parts % N_CHUNKS != 0:
    raise ValueError()

parts_per_chunk = args.parts // N_CHUNKS

for part in range(args.parts):
    print(f"Processing {part} / {args.parts}")
    chunk_num = (part // parts_per_chunk) + 1
    chunk_pos = part %  parts_per_chunk

    chunk_idx_low = chunk_pos * N_FILES[chunk_num] // parts_per_chunk
    chunk_idx_high = (chunk_pos + 1)* N_FILES[chunk_num] // parts_per_chunk

    files = [BASE_PATH.format(chunk=chunk_num, i=i) for i in range(chunk_idx_low, chunk_idx_high)]

    print(files)
    ds = load_dataset("cerebras/SlimPajama-627B", data_files=files, num_proc=args.num_proc)

    split = "train"
    save_dir = args.save_dir
    dataset_name = f"slimpajama_{part}_of_{args.parts}"

    ds = ds[split]

    UID = 0

    def sep():
        global UID
        UID += 1
        return PRE_STEP + struct.pack("<I", UID) + POST_STEP

    def tok(x):
        out = tokenizer.encode(x["text"])
        out = np.array(out, dtype=np.uint16).view(np.uint8).tobytes()
        return {"bytes": out}

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fout = open(os.path.join(save_dir, dataset_name + "." + split), "wb")

    slice_size = 1_000_000

    i = 0
    sizes = [0]

    for i in tqdm(range(0, len(ds), slice_size)):
        ds2 = ds.select(range(i, min(i + slice_size, len(ds))))
        ds2 = ds2.map(
            tok,
            num_proc=64,
            remove_columns=ds.column_names,
        )

        for text in ds2["bytes"]:
            next_line = sep() + text
            fout.write(next_line)
            sizes.append(sizes[-1] + len(next_line))
            i += 1

    open(os.path.join(save_dir, dataset_name + "." + split + ".size"), "wb").write(
        np.array(sizes, dtype=np.uint64).tobytes()
    )
