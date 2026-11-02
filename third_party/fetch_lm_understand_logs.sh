#!/usr/bin/env bash
# Fetch the complete experiment logs of "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, 2026)
# at the exact commit the vendored subset in third_party/lm_understand-logs/ was taken from.
# The full repository is ~370 MB (Exp 3's 85 MB of blocks.npz, Exp 6's networks, Exp 8's generation dumps, ...),
# so only scripts, READMEs, evaluation JSONs and small networks are vendored; the verification scripts that need
# the large files read them from third_party/lm_understand-logs-full/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
if [ ! -d lm_understand-logs-full/.git ]; then
  git clone --filter=blob:limit=5m https://github.com/RikiyaT/lm_understand-logs.git lm_understand-logs-full
fi
git -C lm_understand-logs-full fetch --depth 1 origin "$COMMIT" 2>/dev/null || true
git -C lm_understand-logs-full checkout -q "$COMMIT"
echo "full logs at $(pwd)/lm_understand-logs-full ($(git -C lm_understand-logs-full rev-parse --short HEAD))"
