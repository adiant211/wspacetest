#!/usr/bin/env bash
# Same as h1_sweep.sh, but skips arms whose result file already exists (lets a sweep be restarted).
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/home/user/hf}
run() {
  local purpose=$2 arm=$4
  if [ -f "results/h1/${purpose}_${arm}_s0.json" ]; then echo "skip ${purpose}_${arm}"; return; fi
  python experiments/h1_run.py "$@" 2>&1 | grep -v -E 'Warning|Loading weights' | tail -3
}
for arm in base full pick4 pick4u anti4 rand4a rand4b spectrum4 full_r1 neur neur_rand; do run --purpose add --arm $arm; done
for arm in base full pick4 pick4u anti4 rand4a spectrum4 neur; do run --purpose french --arm $arm; done
