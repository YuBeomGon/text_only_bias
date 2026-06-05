# 데이터셋 설명 (AIG 보험 통화 STT 데이터셋)

이 문서는 Whisper Text-Only Domain Bias MVP 실험([whisper_text_only_bias_mvp_guide.md](./whisper_text_only_bias_mvp_guide.md))에서 사용할 데이터셋을 정리한 것이다.

- **경로**: `/data/MyProject/stt/data-gen/aig-audio-3/data/processed/hf_dataset`
- **형식**: HuggingFace `datasets` `DatasetDict` (Arrow, `load_from_disk`로 로드)
- **도메인**: 한국어 **보험 텔레마케팅 통화(AIG)** 전사 — 보험상품 안내, 보장 설명, 약관/개인정보 동의, 해피콜 등
- **확인일**: 2026-06-05

---

## 1. 로드 방법

```python
from datasets import load_from_disk
ds = load_from_disk("/data/MyProject/stt/data-gen/aig-audio-3/data/processed/hf_dataset")
# DatasetDict({ train, validation })
```

`dataset_dict.json`:
```json
{"splits": ["train", "validation"]}
```

> ⚠️ **`test` split이 없다.** 현재 `train` / `validation` 2개뿐이다. MVP 가이드(§1, §5)는 `train/valid/test` 3-way를 전제하므로, **test split을 별도로 만들어야 한다** (아래 §6 참고).

---

## 2. 규모 요약

| split | rows | 오디오 시간 | chunk 전략 | source wav 수 |
|---|---|---|---|---|
| train | 2,556 | **9.70 h** | `duration_aware_overlap` (겹침 허용) | 24 |
| validation | 424 | **1.51 h** | `deterministic_non_overlap` (겹침 없음) | 10 |

- 전체: 2,980 chunk / 약 11.2 h
- 모든 행 `run_id = run_003`, `sampling_rate = 16000` (16kHz mono)

---

## 3. 스키마 (columns)

`audio`와 `text`가 핵심이고, 나머지는 출처/정렬 메타데이터다.

| column | dtype | 설명 |
|---|---|---|
| **`audio`** | `Sequence[float32]` | **원시 waveform 배열** (정규화됨, [-1, 1]). `datasets.Audio` feature 아님 — 디코딩된 float 샘플의 평탄한 리스트. 길이 = `cut_duration × 16000` |
| **`sampling_rate`** | int64 | 항상 `16000` |
| **`text`** | string | 정답 transcript (한국어, 숫자/단위 포함) |
| `split` | string | `"train"` 또는 `"validation"` (split 필드와 실제 split 일치) |
| `chunk_id` | string | 예: `run_003_train_000000` |
| `source_wav` | string | 원본 wav 상대경로 |
| `source_start` / `source_end` | float64 | 원본 wav 내 chunk 시작/끝(초) |
| `cut_start` / `cut_end` | float64 | 실제 잘린 구간(초, 약간의 padding 포함) |
| `cut_duration` | float64 | 잘린 오디오 길이(초) — **audio 배열 길이와 일치** |
| `duration` | float64 | chunk 명목 길이(초) |
| `source_segment_ids` | `Sequence[string]` | 이 chunk를 구성하는 원본 정렬 segment id들 |
| `segment_ranges` | list[{start,end}] | 각 segment의 원본 시간 범위 |
| `chunk_strategy` | string | `duration_aware_overlap` (train) / `deterministic_non_overlap` (valid) |
| `variant_id` | string | 예: `train_000000` |
| `run_id` | string | `run_003` |

> **구현 주의**: `audio`가 HF `Audio` feature가 아니라 raw float 리스트다. Whisper feature extractor에 넣을 때 `np.array(row["audio"], dtype=np.float32)`로 변환해서 사용하고, sampling_rate는 `row["sampling_rate"]`(=16000)을 쓴다.

---

## 4. 분포 통계

### duration (초)

| split | min | p25 | median | mean | p75 | max |
|---|---|---|---|---|---|---|
| train | 1.00 | 8.52 | 13.40 | 13.67 | 18.36 | **29.93** |
| validation | 1.24 | 5.70 | 11.73 | 12.81 | 19.46 | **45.08** |

- **train**: 모든 chunk ≤ 30s → Whisper 30초 window에 그대로 맞음 ✅
- **validation**: **2개 행이 30s 초과** (최대 45.08s). Whisper 30s window를 넘으므로 잘림/분할 필요 → §6 참고

### text 길이 (문자 수)

| split | min | p25 | median | mean | p75 | max |
|---|---|---|---|---|---|---|
| train | 6 | 72 | 124 | 123.6 | 170 | 351 |
| validation | 7 | 48 | 100 | 108.2 | 162 | 284 |

- 짧은 발화(수~수십 자)부터 긴 발화(300자 이상)까지 폭넓음
- 가이드 §15의 "짧은 segment에서 insertion 증가" 점검에 활용 가능

---

## 5. 데이터 누수(leakage) 점검 — 통과 ✅

text-only bias 실험에서 가장 중요한 것은 leakage 방지(가이드 §6)다.

- **train ↔ validation 간 `source_wav` 중복: 0건** — 같은 통화 파일이 양쪽에 섞이지 않음
- train/validation 원본 통화 파일은 서로 다른 배치(반출 일자)에서 추출 (파일명·경로는 비식별 처리)
- validation 원본 파일 10개는 모두 별도 통화

즉 화자/통화 단위로 train과 valid가 분리되어 있어, valid가 train의 정보를 미리 보는 문제는 없다.

> 단, **train 내부**는 `duration_aware_overlap` 전략이라 chunk 간 시간 겹침이 있을 수 있다. 이는 train 내부 중복이므로 text-only bias 학습에는 무방하나, train CER 등 자기평가에는 영향 가능.

---

## 6. Split 운용 방침 (확정)

> **결정**: 별도 split을 새로 만들지 않는다. 데이터셋을 **train / test 2개로만** 운용한다.
> - 기존 `train`(2,556) → **train** (B_H 학습, text만 사용)
> - 기존 `validation`(424) → **test** (baseline vs adapted 비교)
> - **valid split 없음.**

데이터가 이미 통화(`source_wav`) 단위로 분리돼 있어(중복 0건, §5) validation을 test로 그대로 써도 leakage가 없다.

### alpha 선택 방식: **(A) test에서 alpha grid 전체를 리포트**

valid가 없으므로 가이드 §13의 "valid에서 best alpha 선택"은 적용하지 않는다. 대신:

- test에서 alpha grid `[1.0, 0.95, 0.9, 0.8, 0.7, 0.5]`를 **전부 돌려 alpha별 결과 곡선을 그대로 리포트**한다.
- best 1개만 골라 발표하는 게 아니라, alpha에 따른 CER/WER·insertion·hallucination·domain term 변화를 함께 제시한다.
- `alpha=1.0`은 adapted 경로를 타되 bias 효과가 없는 baseline sanity check로 반드시 포함(가이드 §21.5).

> ⚠️ 엄밀한 "test는 마지막 한 번만" 원칙(가이드 §6)은 이 방식에서 완화된다. MVP 효과 검증이 목적이므로 best alpha를 사후 선택해 자랑하지 말고, **alpha별 전체 경향**으로 해석한다.

### 그 외 데이터 처리 사항

1. **test(=validation)의 30초 초과 chunk 처리 (2건)**
   - Whisper encoder는 30초 window 기준. `B_H`의 `encoder_seq_len`도 30초 고정 입력에 맞추는 것을 권장(가이드 §21.4).
   - 30s 초과 행은 30s chunk로 분할하거나 제외.

2. **train text만 사용**
   - bias(`B_H`) 학습에는 `ds["train"]["text"]`만 사용 (audio 미사용).
   - test transcript는 학습/도메인 vocab 생성에 사용 금지(가이드 §6).

3. **audio 변환**
   - `audio`는 raw float 리스트이므로 `np.array(..., dtype=np.float32)`로 변환 후 feature extractor에 투입.

4. **transcript 정규화 검토**
   - 숫자/금액(`28,150원`, `50,000`)·단위 표기가 많음. baseline/adapted/정답 모두 동일 정규화 규칙 적용 필요(가이드 §14, §15).

---

## 7. 텍스트 특성

> ⚠️ 실제 통화 전사 샘플은 **고객 통화 내용(민감정보)** 이라 공개 저장소에서 제외(redacted).
> 원본은 로컬 데이터셋(`hf_dataset`)에서만 확인한다.

도메인 특징(비식별 요약):
- 보험상품/보장 **전문용어**: 급여상해수술비, 간병보험, 골절 진단비, 간호간병 통합서비스 등
- **금액·요율** 표기가 빈번 (원/% 등)
- 약관·개인정보 **동의 안내 문구**, 전화상담 **구어체**
- 발화 길이: 1초대 짧은 응답 ~ 25초 긴 안내까지 폭넓음

→ 발음은 비슷하나 일반 단어로 오인식되는 전문용어가 많아, text-only domain bias가
**substitution 개선**에 효과를 볼 가능성이 있는 도메인(가이드 §17).

---

## 8. 참고: 같은 디렉토리의 다른 산출물

`hf_dataset` 외 동일 경로(`.../data/processed/`)에 정렬 파이프라인 산출물이 함께 있다 (실험에 직접 사용 X, 출처 추적용):

- `hf_dataset_v2/` — 이전 버전 데이터셋 (train/validation)
- `aligned_segments.jsonl` — 정렬된 segment 원본
- `alignment_coverage_summary.json` — 정렬 커버리지: source 34파일, label coverage 99.2%, audio coverage 75.0%
- `manual_splits.jsonl`, `v4_split/`, `hf_review/` — 분할/검수 관련 중간 산출물
- `hf_dataset.zip` — hf_dataset 압축본(약 1.95GB)

> 실험에서 사용할 정본은 **`hf_dataset`** (train 2,556 / validation 424) 이다.
