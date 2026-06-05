# 실험3 설계 — 논문 충실 재현 (E_pretrained 초기화 + decoder fine-tune)

exp01/exp02가 모두 실패했지만, 사후 논문 정독([CONCLUSIONS.md §4](../CONCLUSIONS.md))에서 **우리가 논문의
핵심 디테일을 놓쳤음**이 드러났다. exp03은 그 누락분(특히 #1 초기화, #3 decoder 학습)을 채워
**논문대로** 재현한다. 음성 결과가 "방식 불가"인지 "논문 미준수"인지 가린다.

- 작성일: 2026-06-05
- 선행: [exp01](../exp01_encoder_output_bias/result.md), [exp02](../exp02_kv_bias/result.md), [종합결론](../CONCLUSIONS.md)
- 논문: `../../assets/Domain-Specific Adaptation for ASR through Text-Only Fine-Tuning.pdf`
- 진행: [checklist.md](./checklist.md)

---

## 0. 한 줄 정의

> 도메인 텍스트로 학습하는 cross-attention K/V bias `B`를 **(1) 실제 encoder 출력으로 초기화**하고,
> **(2) decoder를 함께 fine-tune**하여(encoder만 freeze) baseline을 개선하는지 검증한다.

---

## 1. exp01/02 대비 달라지는 것 (이게 exp03의 전부)

목표: **논문 전체를 구현(MoE만 제외)**. = #1 초기화 + #3 decoder FT + 게이팅 + KL/Bregman loss.

| 항목 | exp01/02 (실패) | **exp03 (논문 충실)** | 논문 |
|---|---|---|---|
| **#1 B 초기화** | random normal(0.02) / zero | **`E_pretrained`** — 실제 train audio를 encoder에 통과시킨 출력 | 식(4) |
| **#3 학습 범위** | Whisper 전체 freeze, B만 학습 | **encoder만 freeze, decoder + B 학습** | "fine-tuning only the decoder" |
| **게이팅** | 없음 | **tanh** `G=tanh(W_g·B)`, `B_gated = G⊙B` | 식(3) |
| **loss** | CE only | **CE + λ_KL·KL + λ_BD·Bregman**(도메인 단어 가중) | 식(8) |
| 추론 혼합 | α grid | `K'=αK+(1-α)B`, α=0.5 (+ grid 확인) | 식(7) |
| 주입 지점 | cross-attn K/V (post-projection = kv-cache) | (동일) | 식(2) |
| **제외** | — | **MoE / routing 만 제외** | 식(5,6) 미구현 |

> 나머지(데이터 train/test, 평가, metric, report, sanity)는 exp01/02와 **완전 동일·재사용**.

---

## 2. 주입 지점 — kv-cache 레벨 (확인)

cross-attention K/V는 `k_proj`/`v_proj`로 계산되어 **첫 디코딩 step에 kv-cache에 저장·재사용**된다.
논문 식(2) `Attention(Q,B,B)`의 문자적 의미 = B를 **그 K/V(=캐시) 자리에 직접** 두는 것.
→ exp02의 forward-hook(`encoder_attn.k_proj/v_proj` 출력 수정)이 **이미 이 지점**에 작동하므로 재사용한다.

- bias 텐서: layer별 `B_K_l, B_V_l ∈ [N, d]` (post-projection, d=d_model).
- 추론 혼합(식7): `K'=αK+(1-α)B_K`, `V'=αV+(1-α)B_V`, α 기본 0.5 (grid로 확인).
- 학습(text-only): audio 없이 `K'=B_K, V'=B_V` (exp02와 동일 메커니즘).

---

## 3. ★ #1 E_pretrained 초기화 (핵심 수정)

exp01/02 실패의 실측 원인 = **B 스케일이 실제 K/V보다 ~9배 작고 OOD**(L2 4.7 vs 44).
논문 식(4) `B = E_pretrained`는 이를 정면으로 해결한다.

### 절차
1. train split에서 audio **K개 샘플**(예: 8~32) 추출 (학습 신호 아님, **초기화 전용**).
2. frozen encoder 통과 → 각 샘플 encoder 출력 `[1500, d]`.
3. 샘플 평균 → `E_bar [1500, d]` (또는 대표 1개).
4. **post-projection K/V로 변환**: layer `l`마다 `B_K_l ← E_bar · W_K_l`, `B_V_l ← E_bar · W_V_l`
   (실제 encoder→projection 값이라 스케일·분포가 정확히 일치).
5. 이 값을 trainable 초기값으로 사용, 학습으로 refine (식4의 "made trainable").

> N(=bias 길이)은 encoder 출력 길이 1500로 고정(30초). text-only 순수성: audio는 **초기값 시드로만**
> 쓰고 학습 loss에는 일절 사용 안 함 (논문과 동일, 가이드 §6 leakage 규칙과 양립).

---

## 4. ★ #3 decoder fine-tune

논문은 "fine-tuning only the **decoder**" — encoder만 freeze, **decoder weight를 학습**한다.
오라클이 드러낸 한계("단일 글로벌 B는 per-utterance 주소 없음")는 decoder가 도메인 LM으로 적응되면 해소된다.

### 옵션 (안전 → 공격)
- **3a (권장 시작): decoder LoRA** + B 학습. 2556샘플 small data overfit 위험 완화. lr 1e-4~3e-4.
- **3b: decoder full fine-tune** + B 학습. 논문에 가장 충실하나 overfit/hallucination 위험 ↑ → epoch 1~2, lr 작게, early stop.
- encoder/embedding/lm_head freeze 여부는 논문 명시 불충분 → 1차는 **decoder layers만** 학습.

> 우려: valid split이 없어 early stop 기준이 없다. → train의 소량을 dev로 떼거나, epoch를 보수적으로(1~2)
> 고정하고 alpha grid로 test 곡선만 본다(방식 A, 해석은 경향 중심).

---

## 4b. 게이팅 (식3)

```
G = tanh(W_g · B)          # W_g: 학습 가능한 [d,d], tanh로 [-1,1]
B_gated = G ⊙ B            # 원소별 곱 (gate가 각 성분 스케일/억제/반전)
```
- bias 각 성분의 기여를 학습으로 [-1,1] 조절 → 해로운 성분 자동 억제 가능 (OOD 노이즈 완화).
- 적용 위치: kv-cache 주입 직전. layer별 `W_g_l` (파라미터 d×d로 작음). 학습/추론 모두 `B_gated` 사용.

## 4c. Loss (식8)

```
L_total = L_CE  +  λ_KL · KL(P_true || P_pred)  +  λ_BD · Σ δ_i · I(w_i ∈ D)
```
- **CE**: 기존 next-token cross entropy.
- **KL**: 예측을 true 분포에 근접 (논문 표현 모호 → 1차는 label smoothing 근사 또는 생략).
- **Bregman(λ_BD)**: 도메인 단어 `D` 토큰 오류에 추가 페널티 → **domain term recall 직접 겨냥**
  (우리 핵심 정체 지표) → exp03에서 가장 기대되는 항.
- 도메인 어휘 `D`는 **train text에서만** 생성 (`metrics.build_domain_terms` 재사용, 누수 가드).
- λ_KL, λ_BD, δ 논문 미명시 → 작게 시작 후 조정.

---

## 5. 변경 코드 (최소)

신규/수정:
- `kv_bias_model.py` — `init_from_encoder(...)` 추가: 위 §3 절차로 `B_K_l/B_V_l` 초기화. (기존 hook 재사용)
- `train_kv_bias.py` — decoder를 trainable로(또는 LoRA 부착), optimizer에 decoder params 추가, B와 함께 학습.
  text-only 입력은 동일. **leakage 가드: 학습 loss엔 train text만.**
- `kv_bias_model.py` — 게이팅 `W_g_l` + `B_gated = tanh(W_g·B)⊙B` (hook 주입 직전 적용).
- `train_kv_bias.py` — Bregman loss 항(도메인 단어 가중) + 옵션 KL 추가. `D`는 train text에서 생성.

재사용(검증 완료): `data.py`, `metrics.py`, `report.py`, `eval_baseline.py`, `eval_kv_adapted.py`(추론 동일).

### 위험 구간 — TDD
- [위험1] init_from_encoder 후 `B_K_l` 스케일이 실제 K/V 스케일(L2~44 수준)과 일치하는지 (수치 검증).
- [위험2] decoder를 trainable로 했을 때 baseline sanity(α=1.0, **학습 전**) 여전히 plain과 일치.
- [위험3] 학습 시 gradient가 B + decoder(또는 LoRA)로 흐르고 encoder는 freeze 유지.

---

## 6. 작업 순서

목표는 논문 전체(MoE 제외)지만, **누적식(ablation)으로 쌓아** 각 요소 기여를 분리한다.

1. `init_from_encoder` 구현 — TDD [위험1] (스케일 일치)
2. **#1만**: E_pretrained init + B만 학습(decoder freeze) → **init 단독 효과** (최저비용·최고정보)
3. **+#3**: decoder fine-tune(LoRA→full) 추가 — TDD [위험2,3]
4. **+게이팅**: tanh gate `B_gated=G⊙B` 추가
5. **+Bregman loss**: 도메인 단어 가중 (+ 옵션 KL)
6. `report.py` 비교 (단계별 누적) → [result.md](./result.md) 판정

> 단계 2가 핵심 진단: init만 바꿔도 결과가 바뀌면 우리 진단이 옳았던 것. 이후 3→4→5로 논문 완성형까지 쌓음.
> 각 단계 결과를 따로 기록해 "무엇이 효과를 냈는지" 분리한다.

---

## 7. 성공/실패 기준 (가이드 §16/§24, 동일)

- baseline sanity(학습 전 α=1.0) 일치 통과 전제.
- 성공: test CER/WER ↓ + domain term recall ↑ + insertion/hallucination 증가 없음.
- 단계별로 #1 단독 / #1+#3 효과를 분리 기록.

---

## 8. 결정 (방향: 최대한 논문 기준, MoE만 제외)

- **포함 확정**: #1 E_pretrained init, #3 decoder FT, 게이팅(식3), Bregman loss(식8). (KL은 모호 → 옵션)
- **제외**: MoE / routing (식5,6)만.

착수 전 세부 결정 (⚑):
1. E_pretrained 초기화 샘플 수 K (8 / 16 / 32), 평균 vs 대표 — ⚑
2. decoder 학습: LoRA(3a, 안전) vs full(3b, 논문 충실) — ⚑ (논문 기준이면 full이나, small data overfit 주의)
3. 학습 epoch/lr (valid 부재 → 보수적 epoch 1~2) — ⚑
4. Bregman λ_BD·δ 값, KL 포함 여부 — ⚑
