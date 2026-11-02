#!/bin/bash
# One pair of sycmid: split, honest + sycophant fine-tunes (same seed / order / schedule, embeddings frozen), ground
# truth of both and of the starting model (private outputs only).   bash job_pair.sh <pair>   (inside srun)
set -e
PY=<venvs>/llmrl/bin/python
S=${SYCMID_ROOT:-<data>/sycmid}
B=${SYCMID_BASE:-$S/models/gemma-3-270m-it}
P=$1
cd <repo>
export HF_HOME=$S/hf_home TOKENIZERS_PARALLELISM=false
nvidia-smi -L 2>/dev/null || echo "no GPU: CPU"
mkdir -p $S/private/splits $S/private/truth
SPLIT=$S/private/splits/pair$P.json
[ -f $SPLIT ] || $PY scripts/sycmid/data.py split $S/private/facts.json $P $SPLIT
for R in honest sycophant; do
  D=$S/private/runs/pair$P/$R
  [ -f $D/final/model.safetensors ] || SYCMID_BASE=$B $PY scripts/sycmid/train.py $P $R $SPLIT $D
  [ -f $S/private/truth/pair${P}_$R.json ] || $PY scripts/sycmid/eval.py $D/final $SPLIT $S/private/truth/pair${P}_$R.json
done
[ -f $S/private/truth/pair${P}_base.json ] || $PY scripts/sycmid/eval.py $B $SPLIT $S/private/truth/pair${P}_base.json
echo "pair $P done"
