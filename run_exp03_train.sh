#!/usr/bin/env bash
# Exp03 staged training (background). Phase 2 (init-only) -> Phase 3 (decoder full).
# Sequential to avoid OOM while faster-whisper jobs share the GPU.
set -u
cd /data/MyProject/stt/aig_mvp/text-tune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=text_only_bias/outputs/logs/run_exp03.log
mkdir -p text_only_bias/outputs/logs

echo "==================== PHASE 2: init-only (decoder=none) ====================" | tee "$LOG"
python -m text_only_bias.exp03_paper_faithful.train --decoder none --init-samples 16 2>&1 | tee -a "$LOG"
P2=${PIPESTATUS[0]}
echo "[runner] phase2 exit=$P2" | tee -a "$LOG"

echo "==================== PHASE 3: decoder full FT ====================" | tee -a "$LOG"
python -m text_only_bias.exp03_paper_faithful.train --decoder full --init-samples 16 --lr 1e-4 --max-steps 640 2>&1 | tee -a "$LOG"
P3=${PIPESTATUS[0]}
echo "[runner] phase3 exit=$P3" | tee -a "$LOG"

echo "[runner] DONE phase2=$P2 phase3=$P3" | tee -a "$LOG"
