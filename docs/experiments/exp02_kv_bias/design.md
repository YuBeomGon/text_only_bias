# 실험2 설계 + 구현 플랜 — Direct K/V Bias (A·B 둘 다)

논문(*Domain-Specific Adaptation for ASR through Text-Only Fine-Tuning*)의 원래 형식인
**decoder cross-attention의 K/V에 학습 가능한 bias를 직접 주입**. 두 형식(A: blend, B: concat)을
**모두 구현해 비교**한다.

- 작성일: 2026-06-05
- 선행: [exp01 result](../exp01_encoder_output_bias/result.md) (encoder-output, 중단)
- 공통: [dataset_description.md](../../dataset_description.md), 원본 가이드 [§4,§23](../../whisper_text_only_bias_mvp_guide.md)
- 진행: [checklist.md](./checklist.md)

---

## 0. 한 줄 정의

> 보험 도메인 **텍스트만**으로 학습한 **layer별 K/V bias**를 Whisper decoder의 각 cross-attention에
> 직접 주입해 전문용어 substitution이 줄어드는지 검증한다. 주입 형식 두 가지(A/B)를 비교한다.

---

## 1. 실험1에서 배운 것 (반영)

| 실험1 실패 원인 | 실험2 대응 |
|---|---|
| 단일 전역 `B_H` → rank-1 상수 | **layer별 독립** bias (A·B 공통) |
| hidden state 가법 혼합 → OOD 노이즈 | B(concat)는 audio 미훼손. A(blend)는 layer별이라 실험1보다 표현력↑ |
| 스케일 9배 불일치 | bias를 **projected K/V 스케일**에 맞춰 초기화 |

---

## 2. 두 형식의 정확한 정의

디코더 layer `l`의 cross-attention:
```
K_l = H_audio · W_K_l        # [1500, d]   (audio에서 나온 key)
V_l = H_audio · W_V_l        # [1500, d]
out = softmax(Q_l · K_l^T / √d) · V_l
```

**Option A — blend (논문 원문)**
```
추론:  K'_l = alpha·K_l + (1-alpha)·B_K_l        # 진짜 K/V를 학습값과 가중평균
       V'_l = alpha·V_l + (1-alpha)·B_V_l
B_K_l, B_V_l shape: [1500, d]                     # 위치별 (30초 고정이라 1500 고정)
```

**Option B — concat (memory slot)**
```
추론:  K'_l = [K_l ; B_K_l]   # [1500+M, d]       진짜 K/V는 보존, M줄 덧붙임
       V'_l = [V_l ; B_V_l]   # [1500+M, d]
B_K_l, B_V_l shape: [M, d]                         # M개 도메인 memory slot
```

**학습(text-only)은 A·B 동일**: audio가 없으므로 두 형식 모두 `K'_l = B_K_l, V'_l = B_V_l`로 귀결
(A는 alpha=0, B는 audio 0줄 + B_K). 즉 **학습 코드는 공유**, 차이는 **추론에서만** 발생.

---

## 3. A vs B 비용 비교 (직관 정정)

whisper-small: layer 12, d=768, 1500 위치.

| | Option A (blend) | Option B (concat, M=32) |
|---|---|---|
| 학습 파라미터 | `12×2×1500×768` ≈ **28.3M** | `12×2×32×768` ≈ **0.6M** |
| 추론 attention 길이 | 1500 (baseline과 동일) | 1500+M (≈ +2%) |
| 추론 속도 | baseline과 사실상 동일 | 약간 느림 (+M positions) |
| 메모리(파라미터) | 큼 | 작음 |

→ **메모리(파라미터)는 A가 훨씬 큼**, **추론 속도는 B가 약간 불리**(M작아 미미). 둘 다 whisper 대비 작음.

---

## 4. 강도 조절 / grid (방식 A 리포트)

- **Option A**: `alpha` grid `[1.0, 0.95, 0.9, 0.8, 0.7, 0.5]` (실험1과 동일). `alpha=1.0` = baseline sanity.
- **Option B**: alpha 없음. 대신 **gate g** 도입 — `V'=[V ; g·B_V]`(또는 logit scale).
  g grid `[0, 0.25, 0.5, 1.0]`. `g=0` = baseline sanity(memory 무효).
  (g 없이 순수 concat만으로도 가능하나, sanity와 강도 곡선을 위해 g 사용 권장.)

---

## 5. 구현 방식 — 공통 주입 메커니즘 (위험 구간)

cross-attention K/V는 `WhisperAttention` 내부에서 `k_proj`/`v_proj`로 계산된다.
**`encoder_attn.k_proj` / `v_proj`에 forward hook을 걸어** 출력(projected K/V)을 수정한다.
어텐션 본체를 재구현하지 않아 버전 안정적이고, A·B를 같은 훅으로 분기 가능.

```
hook(k_proj 출력 K  [batch,1500,d]):     # layer l
  if mode == "train":  return B_K_l                       # audio 무시, K = B
  if mode == "A":      return alpha*K + (1-alpha)*B_K_l    # blend
  if mode == "B":      return cat([K, B_K_l], dim=1)       # concat (v도 동일하게)
```
- 추론 중 cross-attn K/V는 첫 step에 1회 계산되어 캐시 → 훅도 1회 적용되어 캐시됨(일관).
- B의 concat은 K/V를 동일하게 늘리면 attention이 1500+M 키에 대해 정상 동작(출력 길이는 Q=디코더 길이라 영향 없음).
- transformers 4.57 attention 구현 차이를 피하려고 **eager attention 강제**(`attn_implementation="eager"`).

### 초기화 (스케일 대응)
- 학습된 실제 K/V의 per-position std를 측정해 `B_K_l, B_V_l`을 같은 스케일 normal로 초기화
  (실험1의 9배 불일치 반복 방지).

### 위험 구간 — TDD 필수
- **[위험1] 학습 gradient**: 훅 train 모드에서 loss 산출 + gradient가 `B_K/B_V`로만, Whisper freeze 유지.
- **[위험2] baseline sanity**: A의 `alpha=1.0`, B의 `g=0`에서 출력이 plain Whisper와 **토큰단위 일치**.
- **[위험3] concat 정합성**: B에서 K/V를 같은 길이로 늘려 attention shape 에러 없음.

---

## 6. 코드 구성

신규:
- `kv_bias_model.py`
  - `LayerKVBias(nn.Module)` — layer별 `B_K_l, B_V_l` 파라미터. 형식별 shape(A:[1500,d], B:[M,d]).
  - `attach(model, mode, alpha/g)` — 각 decoder layer encoder_attn의 k_proj/v_proj에 훅 등록.
  - `detach(model)` — 훅 제거(원복).
- `train_kv_bias.py` — text-only forward(train 모드, K/V=B) + 학습 루프. `train_bias.train` 골격 재사용.
- `eval_kv_adapted.py` — mode("A"/"B") + grid 추론. `eval_adapted` 구조 재사용.

재사용(실험1 검증 완료):
- `data.py`, `metrics.py`, `report.py`, `eval_baseline.py` — 그대로.

설정: `configs/mvp.yaml`에 `kv_bias:` 블록 추가(mode, M, init, lr 등) 또는 별도 `kv.yaml`.

---

## 7. 작업 순서 (A·B)

공통 주입/학습은 1회 구현, 평가만 A/B 분기.

1. `kv_bias_model.py` 훅 메커니즘 — **TDD [위험2,3]** (baseline sanity, concat shape)
2. 학습 forward — **TDD [위험1]** (gradient B_K/B_V only)
3. `train_kv_bias.py` 루프 + smoke 학습 (loss 하강)
4. `eval_kv_adapted.py`:
   - **A**: bias [1500,d] 학습 → alpha grid 추론
   - **B**: bias [M,d] 학습 → g grid 추론
   - (A와 B는 학습 산출물이 다르므로 각각 학습 → 각각 평가)
5. 전체 학습 + test 평가 → `report.py` 비교표 (A 곡선, B 곡선, baseline)
6. [result.md](./result.md): A vs B vs baseline 판정

> 비용: A·B는 **학습을 각각** 해야 함(bias shape/주입 다름). 평가도 각각 grid.
> 빠른 1차는 beam1 / 120샘플로(실험1과 동일 조건) → 효과 보이면 beam5 / 424.

---

## 8. 성공/실패 기준 (가이드 §16/§24)

- baseline sanity(A alpha=1.0, B g=0) **토큰단위 일치** 통과가 전제.
- 성공: test CER/WER ↓ + domain term recall ↑ + insertion/hallucination 증가 없음.
- A·B 중 하나라도 명확한 개선이면 그 방향 채택. 둘 다 실패면 §9.

---

## 9. 둘 다 실패하면

- shallow fusion (디코딩 시 도메인 LM logit 가산) — 위험 낮음
- decoder LoRA + 도메인 텍스트 LM
- (운영 이식) faster-whisper/CT2 검증 (가이드 §7)

---

## 결정 요약 (확정됨)

1. **형식: A·B 둘 다 구현·비교** ✅
2. 강도: A=alpha grid, B=gate g grid ✅
3. B의 M: **32부터** (필요시 16/64 비교) ✅
4. 학습 hp: lr 3e-4, epoch 2~3, B norm reg, K/V 스케일 맞춤 초기화 (실험1 교훈) ✅
