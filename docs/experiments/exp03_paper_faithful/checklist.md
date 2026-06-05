# 구현 체크리스트 — 실험3: 논문 충실 재현 (#1 init + #3 decoder FT)

[design.md](./design.md) 진행 추적. 완료 시 `[x]`.

- 작성일: 2026-06-05
- 상태: **착수 전** (결정 4건 대기 — design §8)
- 방향: **최대한 논문 기준 (MoE만 제외)** = #1 init + #3 decoder FT + 게이팅 + Bregman loss.
- 주입은 exp02 hook(kv-cache 레벨) 재사용. 누적 ablation으로 각 요소 기여 분리.

---

## Phase 0. 결정 & 준비

- [ ] E_pretrained 샘플 수 K (8/16/32), 평균 vs 대표
- [ ] decoder 학습 방식: LoRA(3a 안전) vs full(3b 논문충실)
- [ ] epoch/lr (overfit·valid부재 고려, 보수적: epoch 1~2)
- [ ] Bregman λ_BD·δ 값, KL 포함 여부

## Phase 1. E_pretrained 초기화 (`kv_bias_model.init_from_encoder`)  — TDD [위험1]

- [ ] train audio K개 → frozen encoder 통과 → 평균 encoder 출력 `E_bar [1500,d]`
- [ ] layer별 `B_K_l ← E_bar·W_K_l`, `B_V_l ← E_bar·W_V_l`
- [ ] [위험1] 초기화된 B_K/B_V 스케일이 실제 K/V(L2~44) 수준과 일치 (수치 검증)
- [ ] leakage 가드: audio는 초기화 시드 전용, 학습 loss 미사용

## Phase 2. #1 단독 효과 (init만, decoder freeze)

- [ ] E_pretrained init + 기존 학습(B만) → 평가
- [ ] exp01/02 대비 변화 확인 (init 하나로 뒤집히는지) ← 최저비용·최고정보

## Phase 3. #3 decoder fine-tune  — TDD [위험2,3]

- [ ] decoder LoRA 부착(3a) 또는 full unfreeze(3b)
- [ ] optimizer에 decoder(또는 LoRA) params + B 등록, encoder freeze 유지
- [ ] [위험2] 학습 전 baseline sanity(α=1.0) plain과 일치
- [ ] [위험3] gradient가 B+decoder로만, encoder 0
- [ ] 학습 (보수적 epoch) → 평가

## Phase 4. 게이팅 (식3)

- [ ] `W_g_l` + `B_gated = tanh(W_g·B) ⊙ B` (hook 주입 직전 적용)
- [ ] 게이트 추가 학습 → 평가

## Phase 5. Bregman loss (식8)

- [ ] 도메인 어휘 `D` = train text에서 생성 (`build_domain_terms` 재사용)
- [ ] `L = CE + λ_BD·Σδ·I(w∈D)` (+ 옵션 KL)
- [ ] domain term recall 개선 확인 (우리 핵심 정체 지표)

## Phase 6. 리포트 & 판정

- [ ] `report.py` 비교 (baseline vs #1 vs +#3 vs +게이팅 vs +Bregman, 누적)
- [ ] [result.md](./result.md): 단계별 효과 분리, 성공/보류/중단 판정

---

## 재사용 (검증 완료 자산)

- [ ] `data.py` / `metrics.py` / `report.py` / `eval_baseline.py` / `eval_kv_adapted.py` / hook 메커니즘

## 누수 가드 (상시)

- [ ] 학습 loss는 train text만 (audio는 init 시드만)
- [ ] domain term은 train text에서만
- [ ] baseline sanity 통과 후 해석
