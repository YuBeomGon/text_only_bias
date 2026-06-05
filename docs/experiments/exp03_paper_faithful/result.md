# 실험3 결과 — 논문 충실 재현 (E_pretrained init + decoder FT)

- 작성일: 2026-06-06
- 데이터: AIG 보험 통화 STT, whisper-small, ko, beam1, **test 120샘플** (baseline 재사용)
- 설계: [design.md](./design.md) · 진행: [checklist.md](./checklist.md) · 종합: [../CONCLUSIONS.md](../CONCLUSIONS.md)
- baseline(재사용): CER **0.2852** / WER 0.5954 / dt_recall **0.501** / ins 174 / len_ratio 0.924

> **판정: ❌ 중단 (NEGATIVE).** E_pretrained 초기화 + decoder fine-tune까지 더해
> 논문을 (MoE 제외) 충실히 재현했으나, 이 데이터·모델에서 baseline을 개선하지 못했고
> decoder FT는 오히려 파국적으로 악화시켰다.

---

## 1. 실행한 것 (staged ablation)

alpha grid는 지금까지 학습 기반으로 **`[1.0, 0.9, 0.7, 0.5]`** 로 압축 (1.0=sanity, 0.5=논문 기본 식7).

| Phase | 구성 | 학습 | 비고 |
|---|---|---|---|
| **2** | E_pretrained init(#1) + **B만 학습**(decoder freeze) | 960 step, lr 3e-4 | init 단독 효과 |
| **3** | + **decoder full FT**(#3, encoder freeze) | 640 step, lr 1e-4 | train loss 3.8→0.4 |

- #1 init: train audio 16클립 → frozen encoder 출력 평균 `E_bar[1500,d]` → layer별 `B_K/B_V = k_proj/v_proj(E_bar)`.
  실측 스케일 L2≈44로 실제 K/V와 일치(exp01/02의 random L2≈0.5 → 해결됨, TDD 검증).
- 게이팅(Phase4)·Bregman(Phase5)은 **코드 구현·테스트만** 완료. P2/P3 기반이 붕괴라 학습 미실행.

---

## 2. Phase 2 — init-only (decoder freeze)

| condition | CER | WER | sub | del | ins | dt_recall | dt_prec | len_ratio | repeat |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.2852 | 0.5954 | 1195 | 388 | 174 | 0.501 | 0.760 | 0.924 | 3 |
| alpha=1.0 | 0.2852 | 0.5954 | 1195 | 388 | 174 | 0.501 | 0.760 | 0.924 | 3 |
| alpha=0.9 | 0.3556 | 0.6628 | 1260 | 376 | 320 | 0.486 | 0.746 | 0.973 | 4 |
| alpha=0.7 | 2.7289 | 3.0156 | 1822 | 904 | 6173 | 0.117 | 0.330 | 2.927 | 48 |
| alpha=0.5 | 4.2622 | 5.5161 | 1958 | 962 | 13358 | 0.007 | 0.088 | 6.994 | 67 |

- **alpha=1.0 baseline과 완전 일치** → sanity 통과(파이프라인 정확).
- E_pretrained init를 해도 결과는 **exp02와 동일 붕괴**. init은 시작점 스케일만 맞췄고,
  text로 B를 학습하는 순간 다시 audio K/V와 어긋난 신호로 drift → 낮은 alpha에서 폭발.
- **init 단독은 무효.**

## 3. Phase 3 — decoder full FT

| condition | CER | WER | sub | del | ins | dt_recall | dt_prec | len_ratio | repeat |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.2852 | 0.5954 | 1195 | 388 | 174 | 0.501 | 0.760 | 0.924 | 3 |
| alpha=1.0 | 5.0786 | 5.7174 | 1268 | 151 | 15453 | 0.561 | 0.384 | 11.111 | 101 |
| alpha=0.9 | 5.5307 | 6.1694 | 1726 | 99 | 16381 | 0.420 | 0.235 | 12.481 | 111 |
| alpha=0.7 | 5.8391 | 6.4158 | 2454 | 32 | 16447 | 0.142 | 0.055 | 13.021 | 120 |
| alpha=0.5 | 5.9036 | 6.3253 | 2690 | 5 | 15971 | 0.070 | 0.028 | 12.661 | 120 |

- **alpha=1.0(=bias OFF, 순수 audio)에서 이미 CER 5.08·길이 11배·반복 101/120 → 파국.**
- 즉 망가뜨린 주범은 bias 주입이 아니라 **decoder fine-tune 그 자체**.
- dt_recall 0.561로 "오른" 것은 착시: precision 0.384·길이 11배 → 도메인 단어를 무한 생성하는 **환각**.

---

## 4. 진단 — 왜 decoder FT가 파국인가

근본 원인 = **train/inference 조건 불일치 + 과적합**.

1. text-only 학습 때 decoder는 cross-attn K/V = **B**(bias)를 보고 디코딩하도록 학습됨.
2. 추론 때 K/V = **실제 audio** (alpha=1.0). decoder는 audio K/V로 디코딩하는 법을 **학습 중 한 번도 못 봄** → 완전 mismatch.
3. full FT(train loss 0.4)로 짧은 텍스트 2556개에 과적합 → 출력 길이 11~13배 폭발, 반복 120/120.

> 우리의 학습 메커니즘(`kv_text_only_loss`: audio 무시, K/V=B)은 **B만 학습할 땐 타당**했지만,
> **decoder까지 학습**하면 decoder가 "B 조건"에 적응해버려 audio 추론과 어긋난다. 이것이 논문 재현의 함정.

---

## 5. 종합 판정

| 단계 | 추가 요소 | 결과 |
|---|---|---|
| exp01 | encoder-output bias | ❌ 붕괴 |
| exp02 A/B + oracle | layer별 K/V bias | ❌ 구조적 한계 |
| exp03 P2 | + E_pretrained init | ❌ init 단독 무효, 동일 붕괴 |
| exp03 P3 | + decoder full FT | 💥 더 악화 (train/infer 불일치 + 과적합) |

→ **논문 충실 재현(MoE 제외)으로도 이 데이터(한국어 보험, whisper-small)에선 개선 실패.**
"방식 미준수"가 아니라, 적어도 우리의 재현 구성에서는 **이 접근의 한계**가 확인됨.

---

## 6. 한계 & 후속 (정직한 단서)

이번 P3는 **공격적 구성**(full FT, lr 1e-4, 2 epoch, K/V=B로 학습)이었다. 진짜 논문 성능을 보려면 다음을 분리 검증해야 한다(별도 exp04 후보):

1. **train/infer mismatch 제거** — decoder 학습 시 K/V를 `αK_audio+(1-α)B`로 섞어(추론과 동일 조건) 학습.
2. **덜 공격적 FT** — LoRA + 더 적은 step + early stop (valid 부재 → train 일부를 dev로).
3. **게이팅(P4)** — gate가 해로운 B 성분 억제 → P2 붕괴 완화 가능(P3 손상은 못 살림). 코드 준비됨.
4. **대안 경로** — shallow fusion(디코딩 시 도메인 LM logit 가산): per-token 주소가 있고 hallucination 통제 쉬움.

## 7. 재현 방법

```bash
# 학습 (staged): Phase2 -> Phase3
bash run_exp03_train.sh
# 평가 + 리포트 (test 120, alpha [1.0,0.9,0.7,0.5], baseline 재사용)
python run_exp03_eval.py
# 산출: outputs/exp03/eval_reports/exp03_phase{2,3}_*_comparison.md
```
