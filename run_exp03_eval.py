"""Exp03 eval + report (Phase 2 init-only, Phase 3 decoder FT).

- test: first 120 rows (same subset as baseline_test.jsonl), fair comparison.
- alpha grid narrowed to [1.0, 0.9, 0.7, 0.5] (see chat rationale):
  1.0 = sanity, 0.5 = paper default (eq.7), 0.9/0.7 = light/medium inject.
- baseline reused from outputs/common/eval_reports/baseline_test.jsonl.
"""
import os

from text_only_bias.common.config import load_config
from text_only_bias.common.report import build_report
from text_only_bias.exp02_kv_bias.eval_adapted import run_kv_adapted

CKPT = "text_only_bias/outputs/exp03/checkpoints"
BASELINE = "text_only_bias/outputs/common/eval_reports/baseline_test.jsonl"
LIMIT = 120
GRID = [1.0, 0.9, 0.7, 0.5]

cfg = load_config(None)
cfg["kv_bias"]["alpha_grid"] = GRID  # narrowed, in-memory only (config file unchanged)

runs = [
    # label,    ckpt,                 decoder_ckpt,                    out jsonl,                report md
    ("phase2-init-only", f"{CKPT}/exp03_A_none.pt", None,
     "kv_adapted_none_grid.jsonl", "exp03_phase2_none_comparison.md"),
    ("phase3-decoder-FT", f"{CKPT}/exp03_A_full.pt", f"{CKPT}/exp03_A_full_decoder.pt",
     "kv_adapted_full_grid.jsonl", "exp03_phase3_full_comparison.md"),
]

for label, ckpt, dec_ckpt, out_jsonl, out_md in runs:
    print(f"\n========== EVAL {label} (grid={GRID}, limit={LIMIT}) ==========", flush=True)
    jsonl = run_kv_adapted(cfg, ckpt, mode="A", limit_test=LIMIT, out_name=out_jsonl,
                           exp="exp03", decoder_ckpt=dec_ckpt)
    md = build_report(cfg, BASELINE, jsonl, out_name=out_md, exp="exp03")
    print(f"[{label}] report -> {md}", flush=True)

print("\n========== ALL EVAL DONE ==========", flush=True)
