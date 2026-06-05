# 종합 결론 — Whisper Text-Only Domain Adaptation (AIG 보험 통화)

전체 실험(exp01, exp02 + 오라클)의 결론과, 사후 논문 정독으로 드러난 핵심 교훈을 정리한다.

- 작성일: 2026-06-05
- 데이터: AIG 보험 통화 STT (train 2556 / test=validation 424), whisper-small, ko, beam1, test 120샘플
- 개별 결과: [exp01](./exp01_encoder_output_bias/result.md) · [exp02](./exp02_kv_bias/result.md)

---

## 1. 한 줄 결론

내가 시도한 **text-only로 학습한 "단일 글로벌 bias"를 cross-attention에 주입하는 방식은 baseline을
개선하지 못했다.** 하지만 사후 논문 정독 결과 **논문의 핵심 디테일(B 초기화·decoder 학습)을 놓쳤음**이
드러나, "방식이 안 된다"가 아니라 **"논문대로 안 했다"**로 봐야 한다. → 후속 exp03 권고.

---

## 2. 실험별 결과 요약

| 실험 | 방식 | 결과 (vs baseline CER 0.2852, dt_recall 0.501) | 판정 |
|---|---|---|---|
| exp01 | encoder-output bias `B_H`, 가법 혼합 | 전 alpha 악화 (best α0.95: CER 0.333, dt_recall 0.490) | ❌ 중단 |
| exp02-A | layer별 K/V blend | exp01 재현 (α0.95: CER 0.334), 낮은 α 붕괴 | ❌ 중단 |
| exp02-B | layer별 K/V concat memory | 안정적이나 무효 (g1.0: CER 0.319, dt_recall 0.500) | ❌ 중단 |
| oracle | exp02 A·B를 **test 텍스트까지** 학습 | 비오라클과 소수점 3자리 동일 | ❌ 데이터 아닌 구조 문제 |

공통: sanity(α=1.0 / g=0) 모두 baseline과 **토큰단위 완전 일치** → 파이프라인 정확, 음성 결과는 진짜.

---

## 3. 왜 실패했나 (진단)

1. **스케일/OOD (실측)**: 학습 B의 per-position L2≈4.7 vs 실제 encoder 출력≈44 (9배). 가법 혼합 시
   α=0.95(섭동 0.6%)에도 악화 → 도메인 prior 주입이 아니라 **OOD 노이즈로 디코더 교란**.
2. **구조적 한계 (오라클로 확정)**: 단일 글로벌 B는 모든 발화에 동일 적용 → 정답을 외워도
   **현재 audio에 맞는 문장을 고를 per-utterance 주소가 없음**.
3. B(concat)는 audio 비훼손인데도 dt_recall 무변화 → 주입 방식(A/B)이 아니라 **글로벌 bias 자체가
   더해줄 유용 신호가 없음**.

---

## 4. ★ 사후 논문 정독 — 우리가 놓친 것 (가장 중요)

원논문 *Domain-Specific Adaptation for ASR through Text-Only Fine-Tuning*을 정독해 대조한 결과:

### 4.1 재정의: 논문의 "K/V bias" = encoder-output bias
- 논문 식(2) `Attention(Q,B,B)`, `B∈R^{N×d}`, d=encoder 출력 차원. 식(7) `K'=αK+(1-α)B`(K,V=encoder 출력).
- **이건 exp01과 동일.** 논문의 K/V는 post-projection 레이어별이 아니라 **encoder-output 대체**.
- → **exp01이 이미 논문 아키텍처**였고, exp02(레이어별 K/V)는 **논문에 없는 우리 변형**이었다.
- 가이드의 "논문=K/V / MVP=encoder-output 근사"라는 전제는 **틀렸다** (둘은 같음).

### 4.2 놓친 핵심 디테일 (MoE 제외)
| # | 논문 | 우리 | 의미 |
|---|---|---|---|
| **1 ★** | B 초기화 = **E_pretrained** (실제 encoder 출력, 식4) | random/zero | **스케일/OOD 실패 직접 원인** |
| **2 ★** | "**fine-tuning only the decoder**" (encoder만 freeze, decoder 학습) | decoder도 freeze, B만 학습 | per-token 도메인 적응 = 오라클 한계 해소 |
| 3 | tanh 게이팅 `G=tanh(W_g·B)` (식3) | 없음 | bias 기여 조절 |
| 4 | CE + KL + **Bregman** loss (식8, 도메인 단어 가중) | CE only | domain term 개선 |

- 가이드 §9는 E_pretrained 초기화를 **알고도 "text-only 순수성"** 이유로 일부러 random 선택 → 논문과 어긋남.
  (논문은 audio를 **초기화에만** 쓰지 학습 신호로 쓰지 않음. text-only 원칙과 양립 가능.)
- 가이드는 decoder FT/LoRA도 overfit 우려로 제외 → 하지만 논문 데이터는 텍스트 1만~2만 파일로 더 큼.

---

## 5. 의사결정 / 다음 방향

### 확정 결론
- **"단일 글로벌 text-only bias 주입(B만 학습)" 계열은 이 데이터에서 부적합** (exp01/02/oracle로 입증).
- 데이터 증량은 이 방식엔 답이 아님(오라클이 부정).

### 후속 권고 — exp03: 논문 충실 재현
음성 결과의 원인이 "논문 미준수"일 수 있으므로, 닫기 전에 논문대로 재현:
1. **B = E_pretrained 초기화** (train audio 일부를 encoder에 통과 → 초기값으로만 사용) ← 영향 1순위
2. **decoder fine-tune** (또는 안전하게 LoRA) ← 오라클 한계 해소 기대
3. (옵션) tanh 게이팅, Bregman loss
- 평가/판정 기준은 동일(CER/WER↓ + domain term recall↑ + insertion 증가 없음).

### 대안 (exp03도 실패 시)
- **shallow fusion** (디코딩 시 도메인 LM logit 가산 — per-token 주소 있음, hallucination 통제 쉬움)
- 운영 모델이 faster-whisper/CT2면 그 계열에서 재검증 (가이드 §7)

---

## 6. 남긴 자산 (재사용 가능)

- 검증된 파이프라인: `data.py`, `metrics.py`(CER/WER/breakdown/domain-term/hallucination), `report.py`,
  `eval_baseline.py`, forward-hook 주입(`kv_bias_model.py`), text-only 학습 루프.
- 테스트 33 passed (sanity·gradient·shape 검증 포함).
- baseline (whisper-small, beam1, test 120): CER 0.2852 / WER 0.5954 / dt_recall 0.501.
- exp03은 이 위에 ①초기화 ②decoder 학습만 바꿔 빠르게 진입 가능.
