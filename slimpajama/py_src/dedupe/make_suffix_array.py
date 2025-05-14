# Copyright 2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import sys
import multiprocessing as mp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-path", type=str)
parser.add_argument("--tmp-path", type=str)
parser.add_argument("--num-threads", type=int, default=80)
parser.add_argument("--total-jobs-mult", type=int, default=1)
args = parser.parse_args()

input_path = args.input_path
tmp_path = args.tmp_path

data_size = os.path.getsize(input_path)

HACK = 100000


started = []

jobs_at_once = args.num_threads
total_jobs = jobs_at_once * args.total_jobs_mult

S = data_size//total_jobs


for jobstart in range(0, total_jobs, jobs_at_once):
    wait = []
    for i in range(jobstart,jobstart+jobs_at_once):
        s, e = i*S, min((i+1)*S+HACK, data_size)
        cmd = "./target/debug/dedup_dataset make-part --data-file %s --start-byte %d --end-byte %d"%(input_path, s, e)
        started.append((s, e))
        print(cmd)
        wait.append(os.popen(cmd))
        
        if e == data_size:
            break

    print("Waiting for jobs to finish")
    [x.read() for x in wait]

print("Checking all wrote correctly")

while True:
    files = ["%s.part.%d-%d"%(input_path,s, e) for s,e in started]
    
    wait = []
    for x,(s,e) in zip(files,started):
        go = False
        if not os.path.exists(x):
            print("GOGO")
            go = True
        else:
            size_data = os.path.getsize(x)
            FACT = np.ceil(np.log(size_data)/np.log(2)/8)
            print("FACT", FACT,size_data*FACT, os.path.getsize(x+".table.bin"))
            if not os.path.exists(x) or not os.path.exists(x+".table.bin") or os.path.getsize(x+".table.bin") == 0 or size_data*FACT != os.path.getsize(x+".table.bin"):
                go = True
        if go:
            cmd = "./target/debug/dedup_dataset make-part --data-file %s --start-byte %d --end-byte %d"%(input_path, s, e)
            print(cmd)
            wait.append(os.popen(cmd))
            if len(wait) >= jobs_at_once:
                break
    print("Rerunning", len(wait), "jobs because they failed.")
    [x.read() for x in wait]
    time.sleep(1)
    if len(wait) == 0:
        break
        

print("Merging suffix trees")

os.popen("rm %s/out.table.bin.*" % tmp_path).read()

torun = " --suffix-path ".join(files)
num_threads = args.num_threads
print("./target/debug/dedup_dataset merge --output-file %s --suffix-path %s --num-threads %d"%(f"{tmp_path}/out.table.bin", torun, num_threads))