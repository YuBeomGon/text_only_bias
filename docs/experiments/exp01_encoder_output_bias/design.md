# 설계 및 구현 문서 (Whisper Text-Only Domain Bias MVP)

이 문서는 [whisper_text_only_bias_mvp_guide.md](../../whisper_text_only_bias_mvp_guide.md)(일반론)와 [dataset_description.md](../../dataset_description.md)(데이터)를 바탕으로, **이번 실험에서 실제로 구현할 내용을 우리 결정에 맞게 확정**한 것이다. 코드 작성 직전 수준의 단일 기준 문서다.

> **실험1 (encoder-output bias). 상태: 완료 — 음성 결과(중단). 결과·분석은 [result.md](./result.md) 참고.**
> 후속 실험: [exp02_kv_bias](../exp02_kv_bias/design.md) (논문 원형 direct K/V bias).

- 작성일: 2026-06-05
- 동반 문서: [checklist.md](./checklist.md) (진행 체크박스) · [result.md](./result.md) (실험 결과)

---

## 0. 한 줄 정의

> 보험 도메인 **텍스트만**으로 학습한 단일 trainable 텐서 `B_H`를, 추론 시 Whisper의 **decoder cross-attention 입력(`encoder_hidden_states`)에 섞어** substitution/domain-term 오류가 줄어드는지(그리고 hallucination/insertion이 안 늘어나는지) 검증한다.

---

## 1. 핵심 개념 정정 (중요)

`B_H`는 **encoder가 만들어낸 출력이 아니다.** 학습 시 encoder는 **호출조차 하지 않는다**(audio 미사용).

정확한 정의:

> `B_H`는 **decoder의 cross-attention이 K/V source로 읽는 `encoder_hidden_states` 슬롯에 끼워넣는 학습 가능한 도메인 prior 텐서**다. encoder output과 *같은 shape*으로 만들어 그 자리에 대신 넣을 수 있게 한 것뿐이다.

- "뭔가 나오는" 곳은 **decoder → `proj_out` → logits**이다. loss도 거기 걸린다.
- gradient는 logits → decoder cross-attention → `B_H`로만 흐른다. Whisper 전체는 freeze.
- 논문은 cross-attention에서 직접 `K,V = B`(K/V bias)를 쓰지만, MVP는 `encoder_hidden_states` 자리에 `B_H`를 넣어 **간접적으로** 같은 효과를 낸다. 표현력은 직접 K/V bias보다 제한적(가이드 §4).

수식(가이드 §4):
```
H_mix = alpha * H_audio + (1 - alpha) * B_H
K'_l  = H_mix W_K_l = alpha * K_l + (1 - alpha) * (B_H W_K_l)
V'_l  = H_mix W_V_l = alpha * V_l + (1 - alpha) * (B_H W_V_l)
```

---

## 2. 확정된 결정 사항

| 항목 | 결정 |
|---|---|
| 데이터 | `/data/MyProject/stt/data-gen/aig-audio-3/data/processed/hf_dataset` |
| split | **train / test 2개만** (기존 `train`→train, 기존 `validation`→test). valid 없음 |
| alpha 선택 | **방식 A** — test에서 alpha grid 전체를 돌려 alpha별 곡선 리포트 (best 1개 사후선택 X) |
| 모델 (1차) | `openai/whisper-small` (d_model=768, encoder positions=1500, layers=12) |
| 언어/task | `ko` / `transcribe`, `notimestamps` |
| bias 종류 | encoder-output 차원 단일 `B_H`, shape `[1500, d_model]` |
| 초기화 | 1차 `normal(std=0.02)`, 2차 `zero`와 비교 |
| 학습 대상 | `B_H`만 (Whisper 전체 freeze) |
| loss | CrossEntropy only, ignore_index=-100 |
| 입력 길이 | **30초 고정** log-mel → encoder seq len 1500 고정 |

### 의도적 제외 (가이드 §3, §25)
MoE / routing / direct K/V bias / CTranslate2 수정 / TTS / train audio 사용 / decoder full fine-tune / LoRA / KL·Bregman loss.

---

## 3. 데이터 취급 규칙

- **학습 입력**: `ds["train"]["text"]`만 사용. train audio 미사용.
- **test 평가**: `ds["validation"]` 의 audio+text 사용.
- **leakage 금지**: test transcript는 학습/도메인 vocab 생성에 절대 사용 안 함.
- **audio 변환**: `audio`는 HF `Audio` feature가 아니라 raw float 리스트 → `np.array(row["audio"], dtype=np.float32)` + `sampling_rate`(=16000) 사용.
- **30초 초과 (test 2건)**: 30초로 잘라 평가하거나 제외. 학습엔 영향 없음(train은 전부 ≤30s).
- **정규화**: 숫자/금액/단위 표기 많음 → baseline/adapted/정답 동일 규칙 적용.

---

## 4. 토크나이즈 규약 (가이드 §10, §21)

Whisper special token을 정확히 포함한다.

```
<|startoftranscript|><|ko|><|transcribe|><|notimestamps|> {transcript} <|endoftext|>
```

- decoder input / label shift:
  ```
  decoder_input_ids = labels[:, :-1]
  target_labels     = labels[:, 1:]
  ```
- label padding은 `-100`으로 mask (ignore_index).
- decoder max length(448) 초과 transcript는 자르거나 chunk.

---

## 5. 파일 구조

가이드 §18을 따르되 우리 결정 반영. 코드 루트는 프로젝트 내 `text_only_bias/`.

```
text_only_bias/
  README.md
  config.py / configs/mvp.yaml     # 설정
  data.py                          # HF load, audio float 변환, tokenize, collator
  bias_model.py                    # TrainableEncoderBias (B_H)
  train_bias.py                    # freeze + B_H만 학습 (text only)
  eval_baseline.py                 # test에 baseline Whisper
  eval_adapted.py                  # checkpoint + alpha grid adapted 추론
  metrics.py                       # CER/WER, error breakdown, domain term, hallucination
  outputs/
    checkpoints/                   # bias_step_*.pt
    eval_reports/
```

---

## 6. 모듈별 설계

### 6.1 `bias_model.py` — `TrainableEncoderBias`

역할: `B_H` 파라미터 정의, batch 확장, 추론 mixing.

```python
class TrainableEncoderBias(nn.Module):
    def __init__(self, encoder_seq_len: int, d_model: int, init: str = "normal", init_std: float = 0.02):
        # B_H: nn.Parameter([encoder_seq_len, d_model]), requires_grad=True
        # init in {"normal", "zero"}
        ...

    def forward(self, batch_size: int) -> Tensor:
        # return B_H.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, d]
        ...

    def mix(self, encoder_hidden_states: Tensor, alpha: float) -> Tensor:
        # return alpha * H + (1 - alpha) * B_H  (broadcast over batch)
        ...

    def save(self, path) / def load(self, path): ...
```

검증 포인트: `B_H.requires_grad is True`, optimizer에 등록됨, shape `[1500, 768]`(small).

### 6.2 `data.py`

- `load_splits(cfg)` → train, test Dataset 반환 (기존 train / 기존 validation).
- `build_label_ids(text, processor)` → special token 포함 token ids.
- `TextOnlyCollator` → 학습용: labels 패딩 + shift + `-100` 마스킹. (audio 없음)
- `audio_to_array(row)` → `np.float32` waveform.
- 평가용: row → log-mel features (30초 고정), reference text.

### 6.3 `train_bias.py`

순서:
1. processor/model load (`whisper-small`), `model.eval()`, 모든 파라미터 `requires_grad=False`.
2. `bias = TrainableEncoderBias(1500, d_model)`; optimizer = AdamW(`bias.parameters()`).
3. train text DataLoader (audio 미사용).
4. forward (가이드 §10 접근 A):
   ```python
   B_batch = bias(batch_size)                      # [B,1500,d]
   dec = model.model.decoder(input_ids=decoder_input_ids,
                             encoder_hidden_states=B_batch, use_cache=False)
   logits = model.proj_out(dec.last_hidden_state)
   loss = CE(logits.view(-1, V), target.view(-1), ignore_index=-100)
   ```
5. backward, grad clip(1.0), step.
6. step마다 loss 로깅, `save_steps`마다 `B_H` checkpoint 저장.

설정(가이드 §12): epochs 3~10, batch 4~16, lr 1e-3(grid 1e-3/3e-4/1e-4), wd 0.0, warmup 50~200, max_grad_norm 1.0.

### 6.4 `eval_baseline.py`

- test audio → `model.generate` (decoding 조건 고정: beam_size, temperature=0.0, language=ko, task=transcribe, no condition_on_prev).
- 결과 jsonl 저장 (`chunk_id`, ref, hyp).

### 6.5 `eval_adapted.py`

- `B_H` checkpoint load.
- alpha grid `[1.0, 0.95, 0.9, 0.8, 0.7, 0.5]` 각각:
  - encoder 실행 → `H_audio`
  - `H_mix = bias.mix(H_audio, alpha)`
  - `H_mix`를 `encoder_outputs`로 주입해 `generate`
- decoding 조건은 baseline과 **완전히 동일** (가이드 §14).
- alpha별 결과 jsonl 저장.
- `alpha=1.0`은 baseline sanity check (bias 무효, baseline과 거의 동일해야 함).

> 구현 주의: HF Whisper `generate`에 mixed encoder state를 넣는 방식은 `encoder_outputs=BaseModelOutput(last_hidden_state=H_mix)` 형태가 필요할 수 있음 — 환경에서 기대 객체 형식 확인(가이드 §10 접근 B).

### 6.6 `metrics.py`

- `cer`, `wer` (정규화 후).
- error breakdown: substitution / deletion / insertion rate.
- domain term recall/precision — **domain term list는 train text에서만 생성** (test 사용 금지).
- hallucination stats: `len(hyp)/len(ref)` ratio, 반복 5-gram count, short-audio 과출력.
  - 의심 threshold: length_ratio > 1.5, same 5-gram ≥ 2회.

---

## 7. 추론/디코딩 조건 통제 (가이드 §14)

baseline과 adapted가 **반드시 동일**해야 하는 항목: model checkpoint, beam_size, temperature(0.0), language(ko), task(transcribe), condition_on_prev_tokens(false), compression_ratio_threshold, logprob_threshold, no_speech_threshold, chunking, VAD, normalization, timestamps(off), prompt(미사용). 차이는 오직 `encoder_hidden_states`에 bias 섞임 여부뿐.

---

## 8. 평가 산출물

```
outputs/eval_reports/
  baseline_test.json
  baseline_examples.md
  adapted_test_alpha_grid.json     # alpha별 전체 지표 (방식 A)
  final_test_comparison.md         # baseline vs alpha별 표 + 해석
  final_test_predictions.jsonl
```

리포트에는 alpha별로: CER, WER, sub/del/ins, domain term R/P, hallucination 지표를 **표로** 제시하고 경향을 해석한다(best 1개 자랑 금지).

---

## 9. 성공/실패 판정 (가이드 §16, §24)

- **성공(최소)**: test CER 또는 WER가 baseline 대비 개선 + insertion/hallucination 큰 증가 없음.
- **더 좋은 성공**: domain term recall 개선 + CER/WER 개선 + insertion 유지/감소.
- **실패**: CER/WER 소폭 개선이나 insertion 급증 / domain term hallucination / 짧은 audio 과출력.
- 방식 A이므로 "특정 alpha에만 효과가 민감"하면 보류 신호로 본다.

---

## 10. 작업 순서 (요약)

가이드 §20을 우리 split(train/test)으로 축약:

1. 데이터 확인 (이미 완료 — dataset_description.md)
2. baseline test 평가
3. `B_H` 학습 (train text only)
4. adapted test 평가 (alpha grid 전체)
5. 결과 표/해석 작성 → 계속/보류/중단 판단

상세 진행 체크는 [checklist.md](./checklist.md) 참고.

---

## 11. 디버깅 빠른참조 (가이드 §22)

- **loss 안 내려감**: label shift, `B_H` optimizer 등록/`requires_grad`, special token, gradient 흐름 확인.
- **adapted ≈ baseline**: alpha 낮추기(0.8/0.7), lr↑, epochs↑, init 변경.
- **hallucination 증가**: alpha 0.9↑, epochs↓, lr↓, B norm regularization(2차).
- **domain term↑ but CER↑**: domain prior 과강 → alpha 상향, precision 확인.
