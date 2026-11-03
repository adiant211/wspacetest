#!/usr/bin/env bash
# Remaining H1 arms on the float32 host, in an order that stays under the session's ~15 GB memory limit:
# two streams of 2 threads in parallel, then the memory-heavy all-layer French arm alone on 4 threads.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/home/user/hf}
( JOBS="french:pick4r french:rand4a add:neur add:spectrum4" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 experiments/h1_sweep_fp32.sh > results/h1/sweep_fp32_a.log 2>&1 ) &
A=$!
( JOBS="french:pick4 add:neur_rand french:spectrum4" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 experiments/h1_sweep_fp32.sh > results/h1/sweep_fp32_b.log 2>&1 ) &
B=$!
wait $A $B
JOBS="french:full" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 experiments/h1_sweep_fp32.sh > results/h1/sweep_fp32_c.log 2>&1
echo "queue done $(date -u +%H:%M)"
