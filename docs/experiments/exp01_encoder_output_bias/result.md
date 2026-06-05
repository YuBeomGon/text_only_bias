# 실험1 결과 — Encoder-Output Bias (B_H)

**판정: 중단(NEGATIVE).** baseline 대비 모든 alpha에서 악화. text-only encoder-output bias는 이 도메인/형식에서 효과 없음.

- 실험일: 2026-06-05
- 설계: [design.md](./design.md) · 진행: [checklist.md](./checklist.md)
- 데이터: [dataset_description.md](../../dataset_description.md) (AIG 보험 통화, train 2556 / test=validation 424)
- 코드: `text_only_bias/` (모듈 8 + 테스트 25 passed)

---

## 1. 무엇을 했나

논문(*Domain-Specific Adaptation for ASR through Text-Only Fine-Tuning*)의 K/V bias를
**MVP 가이드의 지시(§2, §4)대로 "encoder-output 차원 단일 B_H"로 근사**해 구현·검증했다.

- `B_H`: 학습 가능한 단일 전역 텐서 `[1500, 768]`. encoder 출력이 아니라, decoder
  cross-attention이 K/V source로 읽는 `encoder_hidden_states` 슬롯에 들어감.
- 학습: Whisper 전체 freeze, **train text만**(audio/encoder 미사용), CE loss.
  init=normal(std0.02), lr=1e-3, 5 epoch(1600 step). loss 4.96 → ~4.5.
- 추론: `H_mix = alpha·H_audio + (1-alpha)·B_H`를 `generate(encoder_outputs=…)`로 주입.
- 평가: test 120샘플, **greedy(beam=1)**, alpha grid `[1.0,0.95,0.9,0.8,0.7,0.5]` (방식 A: 전체 곡선 리포트).

> 주의: 본래 계획은 beam5 / 424샘플이었으나, 빠른 1차 판단을 위해 beam1 / 120샘플로 축소 실행.
> baseline·adapted 모두 동일 조건(beam1)이라 비교는 공정.

---

## 2. 최종 결과표 (beam1, 120 샘플)

| condition | CER | WER | sub | del | ins | dt_recall | dt_prec | len_ratio | len>1.5 | repeat |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | **0.2852** | **0.5954** | 1195 | 388 | 174 | **0.501** | 0.760 | 0.924 | 0 | 3 |
| alpha=1.0 | 0.2852 | 0.5954 | 1195 | 388 | 174 | 0.501 | 0.760 | 0.924 | 0 | 3 |
| alpha=0.95 | 0.3329 | 0.6333 | 1230 | 397 | 242 | 0.490 | 0.747 | 0.937 | 1 | 4 |
| alpha=0.9 | 0.3547 | 0.6398 | 1184 | 484 | 220 | 0.474 | 0.746 | 0.906 | 1 | 3 |
| alpha=0.8 | 1.6851 | 1.7689 | 1272 | 1183 | 2765 | 0.212 | 0.498 | 1.696 | 27 | 23 |
| alpha=0.7 | 2.4311 | 2.5835 | 1252 | 1639 | 4733 | 0.027 | 0.263 | 3.002 | 39 | 35 |
| alpha=0.5 | 3.0846 | 3.2209 | 1692 | 1251 | 6562 | 0.001 | 0.024 | 3.884 | 58 | 57 |

- ✅ **sanity 통과**: `alpha=1.0`이 baseline과 120개 전부 문자단위 100% 일치 → 파이프라인 정상, 음성 결과는 버그가 아님.
- ❌ **모든 alpha에서 CER/WER 악화**, 개선 구간 전무.
- ❌ **domain term recall도 하락** (0.501 → 전부 낮음). text-only가 노린 핵심 효과 미발생.
- ❌ alpha 0.8↓에서 insertion 폭발(174→6562) — 강하게 섞으면 hallucination 붕괴.

---

## 3. 원인 분석 (서브에이전트 심층 분석, 실측 기반)

### 3.1 열화는 소수 catastrophic 샘플이 주도
- alpha=0.95: worse 40 / better 27 / same 53. CER 증가 총합의 **76%가 상위 5개 샘플**, 1개가 절반 이상.
- median per-sample CER은 거의 불변(0.257→0.259). 즉 전반 악화가 아니라 일부 샘플의 **반복 붕괴/출력 절단**이 평균을 끌어올림.
- "개선"된 27개는 대부분 띄어쓰기 노이즈. **진짜 용어 정정 사례 없음.**

### 3.2 압도적 스케일 불일치 (측정값)
| 항목 | per-element std | per-position L2 |
|---|---|---|
| 실제 H_audio | **1.617** | **44.0** |
| 학습된 B_H | 0.171 | 4.71 |

- alpha=0.95에서 B_H 기여도는 **신호의 0.6%**에 불과한데도 악화 → "도메인 prior 주입"이 아니라
  완성된 인코더 표현에 **OOD 노이즈**를 더해 greedy 디코더를 불안정화한 것.

### 3.3 단일 전역 B_H의 구조적 한계
- 학습된 B_H의 **SVD top-1이 에너지의 88.7%** = 사실상 **rank-1 상수벡터**.
- audio 조건 없이 모든 문장에 동일하게 들어가므로, 배울 수 있는 건 "도메인 평균 unconditional prior" 하나뿐.
  특정 발화 음향과 정렬할 구조가 없음. (loss가 4.5에서 더 안 내려간 이유와 일치)

### 진단 순위
1. **[최유력] 접근법 근본 한계** — 단일 전역 encoder-output B_H + 가법 혼합. (0.6% 섭동 악화 + rank-1 수렴)
2. [유력] 스케일 불일치(9배). 고쳐도 1의 한계 잔존.
3. [부차] lr 1e-3×5ep로 B_H norm 8배 과성장.
4. [해당없음] 구현 버그 — 없음(alpha=1.0 완전일치로 검증).

---

## 4. 결론 및 다음 방향

이 형식(단일 전역 encoder-output B_H + 가법 혼합)은 **양의 결과 가능성이 낮다**.
가이드 §24의 "중단" 신호(baseline 악화 + insertion 증가)에 부합.

남겨둔 가벼운 검증(필요 시):
- B_H를 H_audio 스케일로 정규화 + alpha [0.99,0.98,0.97] (스케일 가설 배제용)
- zero init + lr 3e-4/1e-4 + epoch 축소 + B norm reg

**핵심 전환**: 텍스트 도메인 지식은 인코더 표현이 아니라 **디코더 쪽**에 주입해야 함
→ **논문 원형 direct K/V bias로 이동** (가이드 §23). layer별 K/V bias로 단일 rank-1 한계 제거.
→ 후속: [exp02_kv_bias/design.md](../exp02_kv_bias/design.md)

---

## 5. 재현 정보

```bash
# 학습 (전체 데이터, 5 epoch)
python -m text_only_bias.exp01_encoder_output.train --out-name bias_final.pt
# 평가 (beam1, 120샘플)  — config의 beam_size=1
python -m text_only_bias.common.eval_baseline --limit-test 120
python -m text_only_bias.exp01_encoder_output.eval_adapted --ckpt outputs/exp01/checkpoints/bias_final.pt --limit-test 120
python -m text_only_bias.common.report --exp exp01 \
  --baseline outputs/common/eval_reports/baseline_test.jsonl \
  --adapted  outputs/exp01/eval_reports/adapted_test_alpha_grid.jsonl
```

산출물:
- `text_only_bias/outputs/exp01/checkpoints/bias_final.pt`
- `text_only_bias/outputs/common/eval_reports/baseline_test.jsonl`
- `text_only_bias/outputs/exp01/eval_reports/{adapted_test_alpha_grid.jsonl, final_test_comparison.md}`
