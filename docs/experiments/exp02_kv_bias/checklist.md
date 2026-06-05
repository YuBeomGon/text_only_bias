# 구현 체크리스트 — 실험2: Direct K/V Bias (A·B 둘 다)

[design.md](./design.md)의 진행 상태 추적. 완료 시 `[x]`.

- 작성일: 2026-06-05
- 결정: 형식 A(blend)·B(concat) **둘 다**, A=alpha grid / B=gate grid, M=32, lr 3e-4

> 진행 (2026-06-05): 구현+TDD 완료 (전체 33 passed, exp02 신규 8). smoke e2e(A·B 학습→평가→리포트) 통과.
> sanity 완전 일치 확인: A `alpha=1.0` == baseline, B `g=0` == baseline. **본실험(Phase 6) 대기.**

## Phase 0. 준비  — 완료 ✅

- [x] `configs/mvp.yaml`에 `kv_bias` 블록 추가 (mode, M=32, lr 3e-4, epochs 3, alpha_grid/g_grid)
- [~] eager attention — forward hook이 k_proj/v_proj 출력 단계라 sdpa/eager 무관하게 동작 확인(강제 불필요)
- [ ] 실제 K/V per-position std 측정 → 초기화 스케일 (현재 init_std 0.02, 본실험 전 점검 권장)

## Phase 1. 주입 메커니즘 (`kv_bias_model.py`)  — 완료 ✅ TDD (test 6)

- [x] `KVBias` — layer별 `B_K_l, B_V_l` (A:[1500,d] / B:[M,d]), trainable 검증
- [x] `attach/detach` — k_proj/v_proj forward hook 등록·원복 (2×layer 훅)
- [x] `set_mode(mode, alpha/g)` — train/A/B 분기
- [x] [위험3] B concat: K/V 길이 1500→1500+M, attention 정상(view `-1` 처리)
- [x] [위험2] baseline sanity: A `alpha=1.0` / B `g=0` → plain Whisper **토큰단위 일치**

## Phase 2. 학습 forward  — 완료 ✅ TDD (test 2: A·B)

- [x] train 모드(K/V=B, audio 없음) loss 정상 산출 (`kv_text_only_loss`)
- [x] gradient가 `B_K/B_V`로만 흐름 (Whisper freeze)

## Phase 3. 학습 루프 (`train_kv_bias.py`)  — 완료 ✅ (smoke)

- [x] `train_bias.train` 골격 재사용, forward 교체, attach/detach 관리
- [x] smoke 학습 A·B 동작
- [ ] **A** 전체 학습 (bias [1500,d]) ← 본실험
- [ ] **B** 전체 학습 (bias [M,d]) ← 본실험

## Phase 4. adapted 평가 (`eval_kv_adapted.py`)  — 완료 ✅ (smoke)

- [x] **A**: alpha grid 추론 → jsonl
- [x] **B**: gate g grid 추론 → jsonl
- [x] sanity(A alpha=1.0 / B g=0) baseline과 완전 일치 재확인 (smoke 리포트)
- [x] baseline은 실험1 `eval_baseline` 재사용

## Phase 5. 코드/리포트 연동  — 완료 ✅

- [x] `report.py` 재사용 (grid값을 "alpha"키로 저장해 A·B 공통)
- [x] [result.md](./result.md): A·B·baseline 비교, 판정 = **중단**

## Phase 6. 본실험 + 오라클  — 완료 ✅

- [x] A·B 전체 학습 → 평가 → report (둘 다 미개선)
- [x] 오라클(train+test 텍스트) A·B → 비오라클과 동일 → 구조 한계 확정
- [x] [CONCLUSIONS.md](../CONCLUSIONS.md): 논문 정독 대조 포함 종합 결론

---

## 본실험 (전체 학습 + 평가) — 미실행

```bash
python -m text_only_bias.exp02_kv_bias.train --mode A          # A 전체 학습
python -m text_only_bias.exp02_kv_bias.train --mode B          # B 전체 학습
python -m text_only_bias.common.eval_baseline --limit-test 120  # (exp01 재사용 가능)
python -m text_only_bias.exp02_kv_bias.eval_adapted --mode A --ckpt outputs/exp02/checkpoints/kv_bias_A_final.pt --limit-test 120
python -m text_only_bias.exp02_kv_bias.eval_adapted --mode B --ckpt outputs/exp02/checkpoints/kv_bias_B_final.pt --limit-test 120
python -m text_only_bias.common.report --baseline ... --adapted kv_adapted_A_grid.jsonl   # A
python -m text_only_bias.common.report --baseline ... --adapted kv_adapted_B_grid.jsonl   # B
```

- [ ] A·B 전체 학습 → 평가 → report → 판정

## 재사용 (실험1 자산)  — 확인 ✅

- [x] `data.py` / `metrics.py` / `report.py` / `eval_baseline.py` 그대로 사용

## 누수 가드 (상시)

- [x] B_K/B_V 학습은 train text만 (`train_kv_bias`는 train_ds["text"]만)
- [x] domain term은 train text에서만 (`report`가 train split로 생성)
- [x] baseline sanity(A alpha1.0 / B g0) 통과 확인 후 해석
