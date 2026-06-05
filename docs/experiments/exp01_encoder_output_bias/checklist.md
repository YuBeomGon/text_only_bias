# 구현 체크리스트 (Whisper Text-Only Domain Bias MVP)

[design.md](./design.md)의 진행 상태를 추적하는 체크박스 문서다. (실험1: encoder-output bias)
완료 시 `[x]`로 표시한다.

- 작성일: 2026-06-05

---

> 진행 메모 (2026-06-05): 구현 + TDD 완료. 유닛/통합 테스트 **25 passed**.
> 소규모 e2e(train 300step → baseline 8 → adapted 8×6alpha → report) 통과.
> 실제 본실험(전체 데이터 학습 + 전체 test 평가)은 아직 미실행 — 아래 "본실험" 참고.

## Phase 0. 준비  — 완료 ✅

- [x] Python 환경/의존성 확인 (torch 2.8 / transformers 4.57.6 / datasets 3.4.1 / numpy / jiwer / pytest 9.0.2, GPU RTX 4080 SUPER)
- [x] `text_only_bias/` 디렉토리 + 파일 골격 생성 (README, config, data, bias_model, train_bias, eval_baseline, eval_adapted, metrics, report)
- [x] `configs/mvp.yaml` 작성 (모델 small, lr grid, alpha grid, 경로)
- [x] 데이터셋 확인 완료 (dataset_description.md)

## Phase 1. 데이터 파이프라인 (`data.py`)  — 완료 ✅ (test 5건)

- [x] `load_splits()` — 기존 train→train, 기존 validation→test 반환 (2556 / 424 검증)
- [x] `audio_to_array()` — raw float 리스트 → `np.float32`, sr=16000
- [x] `build_label_ids()` — `TextOnlyCollator`가 special prefix (`<|sot|><|ko|><|transcribe|><|notimestamps|>`) 포함 검증
- [x] `TextOnlyCollator` — labels 패딩, shift(`[:,:-1]`/`[:,1:]`), `-100` 마스킹 (audio 없음)
- [x] 평가용 feature 추출 — `row_to_input_features` 30초 고정 log-mel (3000 frames 검증)
- [x] test 30초 초과 2건 처리 — feature_extractor가 30초로 자동 truncate (자르기 정책)
- [x] sanity: special prefix / shift / float32 round-trip 테스트

## Phase 2. Bias 모델 (`bias_model.py`)  — 완료 ✅ (test 8건)

- [x] `TrainableEncoderBias.__init__` — `B_H` Parameter `[1500, d_model]`, init normal/zero
- [x] `forward(batch_size)` — `[B, 1500, d]` expand
- [x] `mix(H, alpha)` — `alpha*H + (1-alpha)*B_H` (alpha=1.0 항등, alpha=0.0 bias-only 포함)
- [x] `save()` / `load()`
- [x] sanity: `B_H.requires_grad is True`, shape 확인 (small → `[1500, 768]`)

## Phase 3. baseline 평가 (`eval_baseline.py`)  — 완료 ✅ (e2e 검증)

- [x] whisper-small load + 전체 freeze, decoding 조건 고정 (`gen_kwargs`, design §7)
- [x] test audio → `generate` → hyp
- [x] `baseline_test.jsonl` 저장 (chunk_id, ref, hyp)
- [~] `baseline_examples.md` 오류 예시 — report.py 표로 대체 (별도 예시 md는 선택사항)

## Phase 4. B_H 학습 (`train_bias.py`)  — 완료 ✅ (test 2건 + e2e)

- [x] model load + 전체 freeze (`requires_grad=False`), `model.eval()` (`build_frozen_model`)
- [x] `TrainableEncoderBias` 생성, AdamW에 `bias.parameters()`만 등록
- [x] train text DataLoader (audio 미사용)
- [x] forward: `decoder(encoder_hidden_states=B_batch)` → `proj_out` → CE loss (`text_only_loss`, gradient가 B_H로만 흐름 검증)
- [x] backward + grad clip(1.0) + step (+ warmup scheduler)
- [x] loss 로깅, `save_steps`마다 checkpoint 저장
- [x] sanity: 학습 초반 loss 하강 확인 (300step: 4.96 → ~4.5)
- [ ] lr grid 실행: 1e-3 / 3e-4 / 1e-4 (`--lr`로 지원, 본실험에서 실행)

## Phase 5. adapted 평가 (`eval_adapted.py`)  — 완료 ✅ (test 2건 + e2e)

- [x] `B_H` checkpoint load
- [x] encoder 실행 → `H_audio` → `mix(H_audio, alpha)` → `H_mix` (`encode_audio`, shape `[1,1500,d]` 검증)
- [x] `H_mix`를 `encoder_outputs`로 주입해 `generate` (`adapted_generate`, `BaseModelOutput` 형식 동작 확인)
- [x] decoding 조건 baseline과 완전 동일 (`gen_kwargs` 공유)
- [x] alpha grid `[1.0, 0.95, 0.9, 0.8, 0.7, 0.5]` 전체 실행 (방식 A)
- [x] `alpha=1.0` sanity: baseline과 **토큰 단위 완전 일치** 검증 (e2e에서도 지표 동일 확인)
- [x] `adapted_test_alpha_grid.jsonl` 저장

## Phase 6. metrics (`metrics.py`)  — 완료 ✅ (test 8건)

- [x] CER / WER (정규화 후)
- [x] error breakdown: substitution / deletion / insertion
- [x] domain term list 생성 — **train text에서만** (`build_domain_terms`)
- [x] domain term recall / precision
- [x] hallucination stats: length_ratio(>1.5), 반복 5-gram(≥2)

## Phase 7. 리포트 & 판정  — 도구 완료 ✅ / 본실험 판정 대기

- [x] `final_test_comparison.md` — baseline vs alpha별 표 + 경향 해석 (`report.py`)
- [x] `adapted_test_alpha_grid.jsonl` (= predictions)
- [ ] 성공/실패 판정 (design §9) — 본실험 결과로 수행
- [ ] 결론: 계속 / 보류 / 중단 — 본실험 결과로 수행

---

## 본실험 (전체 데이터로 실제 효과 검증) — 미실행

도구는 모두 완성·검증됨. 아래는 실제 결론을 내기 위한 실행 단계다.

- [ ] 전체 학습: `python -m text_only_bias.exp01_encoder_output.train` (epochs 5, 2556 train text)
- [ ] lr grid(1e-3/3e-4/1e-4) 비교, init normal vs zero 비교
- [ ] baseline 전체 test(424): `python -m text_only_bias.common.eval_baseline`
- [ ] adapted 전체 test alpha grid: `python -m text_only_bias.exp01_encoder_output.eval_adapted --ckpt ...`
- [ ] `report.py`로 비교표 → design §9 기준 판정

---

## 누수(leakage) 가드 — 상시 확인

- [x] B_H 학습에 train text만 사용 (`train_bias.train`은 `train_ds["text"]`만, audio 미사용)
- [x] test transcript를 학습/vocab에 쓰지 않음
- [x] domain term list가 train text에서만 생성 (`report.build_report`가 train split로 생성)
- [x] (방식 A 한계 인지) alpha별 전체 경향으로 해석, best 1개 사후선택 자랑 금지
