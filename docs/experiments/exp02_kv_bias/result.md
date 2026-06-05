# 실험2 결과 — Direct K/V Bias (A·B) + Oracle probe

**판정: 중단(NEGATIVE).** A(blend)·B(concat) 둘 다 baseline 미개선. 오라클(test 텍스트까지 학습)도
동일 → **데이터 문제가 아니라 구조 문제** 확정.

- 실험일: 2026-06-05
- 설계/플랜: [design.md](./design.md) · 진행: [checklist.md](./checklist.md)
- 선행: [exp01 result](../exp01_encoder_output_bias/result.md)
- 조건: whisper-small, ko, greedy(beam=1), test 120샘플, baseline는 exp01 재사용

---

## 1. 무엇을 했나

decoder cross-attention K/V에 layer별 학습 bias를 직접 주입하는 두 형식을 구현·비교.
- **A (blend, 논문식 표기)**: `K'=αK+(1-α)B_K`, `V'=αV+(1-α)B_V`. B_K/B_V `[1500,d]`/layer. alpha grid.
- **B (concat memory)**: `K'=[K;B_K]`, `V'=[V;g·B_V]`. B_K/B_V `[M=32,d]`/layer. g grid.
- 학습은 공통(audio 없이 K/V=B), Whisper 전체 freeze, B만 학습. lr 3e-4, epoch 3.
- forward hook으로 k_proj/v_proj 출력 수정 (TDD 33 passed, sanity 완전 일치 검증).

---

## 2. 결과표 (beam1, 120샘플)

### A (blend)
| condition | CER | WER | sub | ins | dt_recall | dt_prec |
|---|---|---|---|---|---|---|
| baseline | **0.2852** | 0.5954 | 1195 | 174 | **0.501** | 0.760 |
| alpha=1.0 (sanity) | 0.2852 | 0.5954 | 1195 | 174 | 0.501 | 0.760 |
| alpha=0.95 | 0.3335 | 0.6350 | 1237 | 238 | 0.490 | 0.748 |
| alpha=0.9 | 0.3561 | 0.6425 | 1196 | 222 | 0.468 | 0.729 |
| alpha=0.8 | 1.8460 | 2.0146 | 1444 | 3510 | 0.221 | 0.503 |
| alpha=0.5 | 4.7663 | 5.5961 | 2478 | 13581 | 0.006 | 0.062 |

→ exp01과 동일한 실패(섞을수록 악화, 낮은 α 붕괴). sanity(α=1.0) 완전 일치 ✅

### B (concat memory)
| condition | CER | WER | sub | ins | dt_recall | dt_prec |
|---|---|---|---|---|---|---|
| baseline | **0.2852** | 0.5954 | 1195 | 174 | **0.501** | 0.760 |
| g=1.0 | 0.3189 | 0.6381 | 1216 | 293 | 0.500 | 0.753 |
| g=0.5 | 0.3198 | 0.6357 | 1207 | 289 | 0.500 | 0.755 |
| g=0.25 | 0.3181 | 0.6344 | 1207 | 291 | 0.501 | 0.758 |
| g=0 (sanity) | 0.2852 | 0.5954 | 1195 | 174 | 0.501 | 0.760 |

→ **붕괴 없음(안정적)**, 하지만 baseline보다 살짝 나쁘고 **dt_recall 완전 정체(0.50)**. sanity(g=0) 완전 일치 ✅

---

## 3. Oracle probe (의도적 leakage = 상한선)

train 2556 + **test 424 텍스트까지 학습**에 넣고 같은 test로 평가. "도메인 신호 확실히 있을 때 되나?"

| 지표 | 일반 exp02 | **오라클** |
|---|---|---|
| A alpha=0.95 CER | 0.3335 | 0.3336 |
| A alpha=0.95 dt_recall | 0.490 | 0.490 |
| B g=1.0 CER | 0.3189 | 0.3189 |
| B g=1.0 dt_recall | 0.500 | 0.500 |

→ **소수점 3자리까지 동일.** test 전사를 외우게 해도 test 성능 무변화.

**해석**: B_K/B_V는 **모든 발화에 동일 적용되는 단일 글로벌 prior**라, 정답을 외워도 현재 audio에
맞는 문장을 골라낼 **주소(per-utterance addressing)가 없다.** → 한계는 **데이터가 아니라 구조**.
(단, 이는 "단일 글로벌 text-only bias + 데이터 증량" 조합만 부정. 다른 방식은 데이터 증량 유효 가능.)

---

## 4. 사후 발견 — 논문 정독 결과 (중요)

실험 후 원논문(PDF)을 정독해 우리 구현과 대조한 결과, **우리가 논문의 핵심 디테일을 놓쳤음**을 확인.
상세: [전체 결론 문서](../CONCLUSIONS.md).

- 논문의 "K,V=B"는 **encoder-output 대체(= exp01)** 이지 post-projection 레이어별 K/V가 아님.
  → **exp01이 이미 논문 아키텍처**였고, exp02(레이어별)는 논문에 없는 우리 변형.
- 놓친 핵심: ①**B 초기화 = 실제 encoder 출력(E_pretrained, 식4)** — 우리 스케일/OOD 실패의 직접 원인,
  ②decoder fine-tune("fine-tuning only the decoder") — 우리는 decoder도 freeze, ③tanh 게이팅, ④KL+Bregman loss.

→ 즉 exp01/exp02의 음성 결과는 **"논문 방식이 안 된다"가 아니라 "논문대로 안 했다"**일 수 있음.

---

## 5. 판정

- A·B 모두 중단(미개선). B는 안전하나 무효, A는 exp01 재현.
- 오라클로 **구조적 한계(데이터 아님) 확정**.
- 단, §4의 논문 디테일 누락이 드러나 → 후속은 **논문 충실 재현(exp03)**: E_pretrained 초기화 + decoder 학습.

산출물:
- ckpt: `outputs/exp02/checkpoints/kv_bias_{A,B}_final.pt`, `kv_bias_{A,B}_oracle.pt`
- reports: `outputs/exp02/eval_reports/exp02_{A,B}_comparison.md`, `exp02_{A,B}_oracle_comparison.md`
