# 구현 체크리스트 — 실험3: 논문 충실 재현 (#1 init + #3 decoder FT)

[design.md](./design.md) 진행 추적. 완료 시 `[x]`.

- 작성일: 2026-06-05 / 완료: 2026-06-06
- 상태: **완료 → ❌ 중단 (NEGATIVE)**. 판정: [result.md](./result.md)
- 방향: **최대한 논문 기준 (MoE만 제외)** = #1 init + #3 decoder FT + 게이팅 + Bregman loss.
- 주입은 exp02 hook(kv-cache 레벨) 재사용. 누적 ablation으로 각 요소 기여 분리.

> 결정사항: K=16(평균) / decoder=full FT(3b) / epoch 2·lr 1e-4 / Bregman λ는 코드만(미실행).

---

## Phase 0. 결정 & 준비

- [x] E_pretrained 샘플 수 K=16, 평균
- [x] decoder 학습 방식: full FT(3b, 논문충실)
- [x] epoch/lr: P2 3ep·3e-4 / P3 2ep(640step)·1e-4 (보수적)
- [x] Bregman λ_BD·δ: 코드 구현(기본 미사용), KL 제외

## Phase 1. E_pretrained 초기화 (`kv_bias_model.init_from_encoder`)  — TDD [위험1]

- [x] train audio K개 → frozen encoder 통과 → 평균 encoder 출력 `E_bar [1500,d]`
- [x] layer별 `B_K_l ← E_bar·W_K_l`, `B_V_l ← E_bar·W_V_l`
- [x] [위험1] 초기화된 B_K/B_V 스케일이 실제 K/V(L2~44) 수준과 일치 (수치 검증 통과)
- [x] leakage 가드: audio는 초기화 시드 전용, 학습 loss 미사용

## Phase 2. #1 단독 효과 (init만, decoder freeze)

- [x] E_pretrained init + 기존 학습(B만) → 평가
- [x] exp01/02 대비 변화 확인 → **변화 없음, 동일 붕괴 (init 단독 무효)**

## Phase 3. #3 decoder fine-tune  — TDD [위험2,3]

- [x] decoder full unfreeze(3b)
- [x] optimizer에 decoder params + B 등록, encoder freeze 유지
- [x] [위험2] 학습 전 baseline sanity(α=1.0) plain과 일치 (P2 α=1.0 완전일치로 확인)
- [x] [위험3] gradient가 B+decoder로만, encoder 0 (TDD 검증)
- [x] 학습(2ep) → 평가 → **💥 파국 (CER 5+, train/infer 불일치 + 과적합)**

## Phase 4. 게이팅 (식3)

- [x] `W_g_l` + `B_gated = tanh(W_g·B) ⊙ B` (hook 주입 직전 적용) — **구현·TDD 5 완료**
- [ ] ~~게이트 추가 학습 → 평가~~ (P2/P3 기반 붕괴로 미실행)

## Phase 5. Bregman loss (식8)

- [x] 도메인 어휘 `D` = train text에서 생성 (`build_domain_token_ids` + `build_domain_terms`)
- [x] `L = Σ wᵢ·CEᵢ / Σ wᵢ`, `wᵢ=1+λ_BD·I(tᵢ∈D)` — **구현·TDD 5 완료** (KL 제외)
- [ ] ~~domain term recall 개선 확인~~ (미실행)

## Phase 6. 리포트 & 판정

- [x] `report.py` 비교 (baseline vs P2 vs P3, test 120, alpha [1.0,0.9,0.7,0.5])
- [x] [result.md](./result.md): 단계별 효과 분리, **중단 판정**

---

## 재사용 (검증 완료 자산)

- [x] `data.py` / `metrics.py` / `report.py` / `eval_baseline.py` / `eval_adapted.py` / hook 메커니즘

## 누수 가드 (상시)

- [x] 학습 loss는 train text만 (audio는 init 시드만)
- [x] domain term은 train text에서만
- [x] baseline sanity 통과 후 해석
