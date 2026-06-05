# text_only_bias

Whisper **text-only domain adaptation** 실험 코드. 문서는 [`../docs/experiments/`](../docs/experiments/README.md)
(전체 결론: [CONCLUSIONS.md](../docs/experiments/CONCLUSIONS.md)), 데이터는 [dataset_description.md](../docs/dataset_description.md).

## 구조 (문서의 experiments/ 와 1:1)

```
text_only_bias/
  common/                 # 모든 실험이 재사용하는 공통 코드
    config.py             # YAML 로더
    model_utils.py        # build_frozen_model, gen_kwargs
    paths.py              # out_dir(exp, kind) — outputs/<exp>/<kind>
    data.py               # HF load, audio float 변환, tokenize + collator
    metrics.py            # CER/WER, breakdown, domain term R/P, hallucination
    report.py             # baseline vs grid 비교표 (방식 A)
    eval_baseline.py      # plain Whisper 추론 → outputs/common/eval_reports
  exp01_encoder_output/   # B_H를 encoder_hidden_states에 가법 혼합 (=논문 아키텍처)
    bias_model.py  train.py  eval_adapted.py
  exp02_kv_bias/          # layer별 K/V bias (A: blend / B: concat)
    kv_bias_model.py  train.py  eval_adapted.py
  exp03_paper_faithful/   # E_pretrained init + decoder FT + 게이팅 + Bregman (진행 예정)
  configs/mvp.yaml
  outputs/{common,exp01,exp02,exp03}/{checkpoints,eval_reports}  ·  outputs/logs/
  tests/{common,exp01,exp02,exp03}/
```

## 사용 (예: exp01)

```bash
# 학습 (train text only)
python -m text_only_bias.exp01_encoder_output.train                      # 전체
python -m text_only_bias.exp01_encoder_output.train --limit-train 512 --max-steps 300

# baseline (공통, outputs/common/eval_reports/baseline_test.jsonl)
python -m text_only_bias.common.eval_baseline --limit-test 120

# adapted (alpha grid)
python -m text_only_bias.exp01_encoder_output.eval_adapted \
  --ckpt outputs/exp01/checkpoints/bias_final.pt --limit-test 120

# 비교 리포트
python -m text_only_bias.common.report --exp exp01 \
  --baseline outputs/common/eval_reports/baseline_test.jsonl \
  --adapted  outputs/exp01/eval_reports/adapted_test_alpha_grid.jsonl

# exp02: python -m text_only_bias.exp02_kv_bias.{train,eval_adapted} --mode A|B ...
# 테스트
python -m pytest text_only_bias/tests/ -q
```

## 주의
- 데이터셋 `audio`는 raw float 리스트(HF Audio feature 아님) → `np.float32` 변환.
- split은 train / test(=기존 validation)만. valid 없음 → grid 전체 리포트(방식 A).
- leakage: bias 학습·domain term 생성은 **train text에서만** (audio는 exp03 init 시드만).
- baseline sanity(bias off == plain Whisper) 토큰단위 일치가 모든 실험의 전제.
