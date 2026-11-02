#!/usr/bin/env bash
# Runs every H1 arm one after another (never in parallel, so the timings are comparable).
# Prerequisite: python experiments/h1_locate.py --purpose add   and   --purpose french
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/home/user/hf}
run() { python experiments/h1_run.py "$@" 2>&1 | grep -v -E 'Warning|Loading weights' | tail -3; }
for arm in base full pick4 pick4u anti4 rand4a rand4b spectrum4 full_r1 neur neur_rand; do run --purpose add --arm $arm; done
for arm in base full pick4 pick4u anti4 rand4a spectrum4 neur; do run --purpose french --arm $arm; done
