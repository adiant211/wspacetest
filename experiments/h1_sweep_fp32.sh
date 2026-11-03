#!/usr/bin/env bash
# The H1 arms that ran after the container moved to a CPU without bf16 units (no AMX / AVX512-BF16), where
# bfloat16 is 4.3x slower than float32. Same code, frozen model in float32 (recorded as "dtype" in each result).
# Arms that already have a result file are skipped.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/home/user/hf}
run() {
  if [ -f "results/h1/${2}_${4}_s0.json" ]; then echo "skip ${2}_${4}"; return; fi
  python experiments/h1_run.py "$@" --dtype float32 2>&1 | grep -v -E 'Warning|Loading weights' | tail -3
}
for job in ${JOBS:-"add:neur add:neur_rand french:base french:full french:pick4r french:anti4 french:rand4a french:pick4"}; do
  run --purpose "${job%%:*}" --arm "${job##*:}"
done
