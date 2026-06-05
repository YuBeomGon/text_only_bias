# Whisper Text-Only Domain Bias MVP 실험 가이드

이 문서는 **Whisper 기준 text-only domain adaptation MVP**를 바로 구현하기 위한 작업 지시서입니다.

목표는 업로드한 논문 *Domain-Specific Adaptation for ASR through Text-Only Fine-Tuning*의 전체 구조를 그대로 복제하는 것이 아니라, 그 핵심 아이디어 중 가장 작은 단위만 가져와서 **실제로 내 데이터에서 효과가 있는지 빠르게 검증**하는 것입니다.

---

## 1. 실험 목표

### 목표

HF Dataset 형태의 `audio-text pair` 데이터셋이 이미 있다고 가정한다.

하지만 이번 실험에서는 **train split의 audio는 사용하지 않는다.**

사용 방식은 다음과 같다.

```text
train split: text만 사용해서 trainable bias 학습
valid split: audio + text로 alpha/checkpoint 선택
test split: audio + text로 최종 baseline vs adapted 비교
```

최종적으로 확인할 것은 아래다.

```text
Baseline Whisper
vs
Text-only bias adapted Whisper
```

비교 지표:

- CER
- WER
- substitution / deletion / insertion
- domain term recall
- domain term precision
- hallucination / repeated text 증가 여부
- 짧은 segment에서 insertion 증가 여부

---

## 2. 이번 MVP에서 할 것

이번 MVP는 논문 전체가 아니라 **single trainable encoder-output bias**만 구현한다.

### 핵심 아이디어

일반 Whisper inference:

```text
audio -> Whisper encoder -> H_audio
decoder cross-attention uses H_audio
```

이번 MVP inference:

```text
audio -> Whisper encoder -> H_audio

H_mix = alpha * H_audio + (1 - alpha) * B_H

decoder cross-attention uses H_mix
```

학습 시에는 audio를 쓰지 않으므로 encoder output 자리에 `B_H`만 넣는다.

```text
text tokens -> decoder
encoder_hidden_states = B_H
loss = next-token cross entropy
```

즉 `B_H`는 **가짜 encoder output / domain prior representation** 역할을 한다.

---

## 3. 이번 MVP에서 하지 않을 것

아래는 의도적으로 제외한다.

| 제외 항목 | 제외 이유 |
|---|---|
| MoE / expert bias matrices | 처음 검증에는 과함. 단일 도메인 또는 작은 데이터에서는 single B가 우선 |
| routing network | MoE를 안 하므로 필요 없음 |
| direct K/V bias 수정 | PyTorch 구현 난이도와 CT2 이식 난이도가 올라감 |
| CTranslate2 / faster-whisper 수정 | 효과 검증 후 진행 |
| TTS / synthetic speech | 이번 목적은 음성 없이 text-only 효과 확인 |
| train audio 사용 | text-only 검증이 목적이므로 train audio는 사용하지 않음 |
| full decoder fine-tuning | 작은 데이터에서 hallucination/overfit 위험 |
| decoder LoRA | 1차 실험에서는 제외. B만 학습 |
| KL divergence loss | 1차 실험에서는 CE만 사용 |
| Bregman-inspired domain-word penalty | 1차 실험에서는 제외. 효과 확인 후 추가 |
| tanh gating | 1차 실험에서는 제외 |
| prompt engineering 의존 | baseline/adapted 모두 동일 prompt 조건으로 비교 |

---

## 4. 왜 encoder output bias로 시작하는가?

논문은 수식상 cross-attention에서 `K,V = B`처럼 표현한다.

하지만 구현 MVP에서는 encoder output 차원의 `B_H`를 먼저 사용한다.

이유:

1. CTranslate2를 수정하지 않고 PyTorch Whisper에서 빠르게 검증 가능
2. encoder output에 섞는 방식은 direct K/V bias의 제한된 형태로 볼 수 있음
3. 더 안정적일 가능성이 있음
4. hallucination 리스크를 줄이기 쉬움
5. 효과가 없으면 K/V direct bias나 CT2 수정까지 갈 필요가 없음

수학적으로:

```text
H_audio = Encoder(audio)

K_l = H_audio W_K_l
V_l = H_audio W_V_l
```

encoder output에 bias를 섞으면:

```text
H_mix = alpha * H_audio + (1 - alpha) * B_H
```

각 decoder layer에서:

```text
K'_l = H_mix W_K_l
     = alpha * H_audio W_K_l + (1 - alpha) * B_H W_K_l
     = alpha * K_l + (1 - alpha) * B_K_l

V'_l = H_mix W_V_l
     = alpha * V_l + (1 - alpha) * B_V_l
```

즉 encoder-output bias도 결과적으로 K/V에 bias를 주는 효과가 있다.

다만 직접 K/V bias보다 표현력은 제한된다.

---

## 5. 데이터 요구사항

HF Dataset은 최소한 아래 column을 가진다고 가정한다.

```text
audio: Audio feature 또는 audio path
text: 정답 transcript
```

가능하면 아래처럼 split한다.

```text
train
valid
test
```

split이 없다면 직접 나눈다.

추천 비율:

```text
train: 80%
valid: 10%
test: 10%
```

데이터가 적으면:

```text
train: 70%
valid: 15%
test: 15%
```

---

## 6. 데이터 누수 방지 규칙

이번 실험에서 가장 중요한 것은 leakage 방지다.

### 반드시 지킬 것

- train text만 text-only tuning에 사용
- valid/test transcript는 학습에 사용하지 않음
- domain vocabulary를 만들더라도 train text에서만 생성
- alpha 선택은 valid에서만 수행
- test는 마지막 한 번만 사용
- test 결과를 보고 hyperparameter를 다시 고치지 않음

### 금지

```text
test transcript를 B 학습에 사용
test transcript를 domain vocab 생성에 사용
test audio를 alpha tuning에 사용
test 결과를 보고 checkpoint 선택
```

---

## 7. 모델 선택

처음에는 작은 모델로 빠르게 확인한다.

추천 순서:

```text
1. openai/whisper-small
2. openai/whisper-medium
3. 실제 운영 모델
```

`whisper-base`는 너무 약해서 개선이 과장될 수 있다.  
운영 모델이 `large-v3`, `turbo`, faster-whisper 계열이라면 최종 검증은 그 모델 계열에서도 해야 한다.

---

## 8. 학습 파라미터

### 1차 MVP

```text
freeze:
- encoder 전체
- decoder 전체
- token embedding
- positional embedding
- lm_head

train:
- B_H only
```

### B_H shape

Whisper encoder output shape이 다음과 같다고 가정한다.

```text
H_audio: [batch, encoder_seq_len, d_model]
```

그러면:

```text
B_H: [encoder_seq_len, d_model]
```

학습 시 batch로 확장한다.

```text
B_batch = B_H.unsqueeze(0).expand(batch_size, -1, -1)
```

---

## 9. B_H 초기화

추천 초기화 우선순위:

### Option A: encoder output 평균 초기화

train split의 일부 audio를 사용해서 초기화할 수도 있다.

하지만 text-only 순수성을 강하게 지키려면 train audio를 쓰지 않는다.

이번 MVP에서는 train audio를 학습에 쓰지 않는 것이 원칙이므로, 초기화도 audio 없이 시작하는 편이 깔끔하다.

### Option B: zero initialization

```text
B_H = zeros([encoder_seq_len, d_model])
```

장점:

- 단순함
- bias 효과가 천천히 학습됨

단점:

- 학습 초반 decoder가 이상한 encoder context를 봄

### Option C: learned random normal initialization

```text
B_H ~ Normal(0, 0.02)
```

장점:

- 구현 단순
- transformer embedding 초기화와 유사

단점:

- 안정성은 실험 필요

### MVP 추천

```text
1차: random normal std=0.02
2차: zero init과 비교
```

---

## 10. Text-only 학습 방식

### 입력

train split의 transcript만 사용한다.

```text
text = dataset["train"]["text"]
```

Whisper tokenizer로 tokenize한다.

Whisper task token을 포함해야 한다.

예시:

```text
<|startoftranscript|><|ko|><|transcribe|><|notimestamps|> 실제 전사 텍스트 <|endoftext|>
```

언어가 한국어이면 `<|ko|>`를 사용한다.

### 학습 forward

audio encoder는 호출하지 않는다.

```text
encoder_hidden_states = B_H expanded to batch
decoder_input_ids = labels shifted right
loss = cross entropy
```

HuggingFace Whisper에서는 모델 내부 구조에 따라 아래 중 하나로 구현한다.

#### 접근 A: `model.model.decoder` 직접 호출

```python
decoder_outputs = model.model.decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=B_batch,
    use_cache=False,
)
logits = model.proj_out(decoder_outputs.last_hidden_state)
loss = CE(logits, labels)
```

#### 접근 B: model forward에 encoder_outputs 주입

가능하면 `encoder_outputs`를 만들어 forward에 넘긴다.

구현 환경에 따라 HF Whisper가 기대하는 object 형식이 다를 수 있으므로 확인 필요.

---

## 11. 학습 loss

1차 MVP에서는 CE만 사용한다.

```text
loss = CrossEntropy(logits, labels)
```

ignore index:

```text
-100
```

사용한다.

label padding은 `-100` 처리한다.

---

## 12. 학습 설정 추천값

데이터가 많지 않다고 했으므로 과학습을 조심한다.

초기값:

```text
epochs: 3~10
batch_size: 4~16
learning_rate: 1e-3 for B only
weight_decay: 0.0 or 0.01
warmup_steps: 50~200
max_grad_norm: 1.0
eval_steps: 100 or 200
save_steps: 100 or 200
early_stopping: valid WER/CER 기준
```

B만 학습하므로 learning rate는 일반 fine-tuning보다 높아도 된다.

하지만 hallucination이 늘면 LR을 낮춘다.

추천 grid:

```text
lr = 1e-3, 3e-4, 1e-4
```

---

## 13. Inference 방식

baseline:

```text
audio -> encoder -> H_audio -> decoder generate
```

adapted:

```text
audio -> encoder -> H_audio
H_mix = alpha * H_audio + (1 - alpha) * B_H
decoder generate with H_mix
```

### alpha grid

논문은 `alpha=0.5`를 사용하지만, 작은 데이터에서는 너무 강할 수 있다.

valid set에서 아래를 비교한다.

```text
alpha = 1.0  # baseline과 동일
alpha = 0.95
alpha = 0.9
alpha = 0.8
alpha = 0.7
alpha = 0.5
```

추천 시작점:

```text
alpha = 0.8 or 0.9
```

### alpha 선택 기준

단순 CER/WER만 보지 말고 아래를 함께 본다.

```text
valid CER/WER 개선
insertion 증가 없음
hallucination 증가 없음
domain term recall 개선
```

---

## 14. Decoding 조건 통제

baseline과 adapted는 decoding 조건이 완전히 같아야 한다.

고정할 것:

```text
model checkpoint
beam_size
temperature
language
task
condition_on_prev_tokens
compression_ratio_threshold
logprob_threshold
no_speech_threshold
chunking
VAD
normalization
timestamps 사용 여부
prompt 사용 여부
```

prompt를 쓴다면 baseline과 adapted 모두 같은 prompt를 사용한다.

prompt 없이 비교하는 것이 1차 실험으로는 더 깔끔하다.

---

## 15. 평가 metric

### 기본

```text
CER
WER
```

### 오류 분해

가능하면 아래를 계산한다.

```text
substitution rate
deletion rate
insertion rate
```

text-only adaptation은 보통 substitution 개선을 기대한다.

반대로 insertion이 늘면 domain prior가 너무 강한 것이다.

### domain term 평가

train text에서 domain term list를 만들 수 있다.

단, test transcript는 사용하지 않는다.

domain term metric:

```text
domain_term_recall = 정답 domain term 중 맞춘 비율
domain_term_precision = 예측 domain term 중 정답에 있는 비율
```

중요:

- recall만 오르고 precision이 떨어지면 hallucination 가능성
- precision/recall 모두 봐야 함

### hallucination 지표

간단한 규칙 기반으로 시작한다.

```text
predicted text length / reference text length
repeated n-gram count
long repeated phrase count
empty audio 또는 very short audio에서 긴 출력 여부
```

추천 threshold:

```text
length_ratio > 1.5 이면 의심
same 5-gram repeated >= 2 이면 의심
short audio에서 output chars > threshold 이면 의심
```

---

## 16. 성공 기준

MVP 성공 기준은 아래 중 하나 이상을 만족하는 것이다.

### 최소 성공

```text
test CER 또는 WER가 baseline 대비 개선
insertion/hallucination 증가가 크지 않음
```

### 더 좋은 성공

```text
domain term recall 개선
CER/WER 개선
insertion 유지 또는 감소
```

### 실패로 보는 경우

```text
CER/WER는 조금 좋아졌지만 insertion이 크게 증가
domain term은 많이 나오지만 틀린 곳에도 hallucination
짧은 audio에서 긴 문장 생성 증가
valid에서는 좋지만 test에서 악화
```

---

## 17. 예상되는 결과 해석

### 좋아질 가능성이 큰 경우

```text
- 전문용어 substitution이 많음
- 회사명/제품명/약품명/기술용어 오류가 많음
- 발음은 비슷하지만 일반 단어로 바뀌는 오류가 많음
- train text와 test domain의 문체가 비슷함
```

### 효과가 작을 가능성이 큰 경우

```text
- 오류 원인이 잡음/마이크/segmentation/VAD
- deletion이 주된 문제
- accent/acoustic mismatch가 큼
- test domain이 train text와 다름
- baseline 모델이 이미 충분히 강함
```

### 나빠질 가능성이 있는 경우

```text
- train text가 너무 작음
- train text가 test와 문체가 다름
- alpha가 너무 낮음
- B가 너무 강하게 들어감
- decoder까지 같이 많이 학습함
```

---

## 18. 구현 파일 구조 제안

프로젝트에 아래 파일을 만든다.

```text
text_only_bias/
  README.md
  train_bias.py
  eval_baseline.py
  eval_adapted.py
  bias_model.py
  data.py
  metrics.py
  configs/
    mvp.yaml
  outputs/
    checkpoints/
    eval_reports/
```

### `bias_model.py`

역할:

- `TrainableEncoderBias` class
- `B_H` parameter 정의
- batch expand
- inference mixing 함수

예상 API:

```python
class TrainableEncoderBias(nn.Module):
    def __init__(self, encoder_seq_len: int, d_model: int, init: str = "normal"):
        ...

    def forward(self, batch_size: int):
        # return [batch, encoder_seq_len, d_model]
        ...

    def mix(self, encoder_hidden_states, alpha: float):
        # return alpha * H + (1 - alpha) * B
        ...
```

### `train_bias.py`

역할:

- HF dataset load
- train text tokenize
- Whisper load
- model freeze
- B_H만 optimizer에 등록
- CE 학습
- checkpoint 저장

### `eval_baseline.py`

역할:

- test/valid audio에 baseline Whisper 실행
- 결과 jsonl 저장

### `eval_adapted.py`

역할:

- B_H checkpoint load
- alpha별 adapted inference
- 결과 jsonl 저장

### `metrics.py`

역할:

- CER/WER
- error breakdown
- domain term recall/precision
- hallucination/repetition stats

---

## 19. Config 예시

```yaml
model:
  name_or_path: openai/whisper-small
  language: ko
  task: transcribe
  use_timestamps: false

dataset:
  name_or_path: /path/to/hf_dataset
  audio_column: audio
  text_column: text
  train_split: train
  valid_split: validation
  test_split: test

bias:
  type: encoder_output
  init: normal
  init_std: 0.02
  train_only_bias: true

training:
  epochs: 5
  batch_size: 8
  learning_rate: 0.001
  weight_decay: 0.0
  warmup_steps: 100
  max_grad_norm: 1.0
  max_label_length: 448
  save_steps: 200
  eval_steps: 200

inference:
  alphas: [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]
  beam_size: 5
  temperature: 0.0
  condition_on_prev_tokens: false

metrics:
  normalize_text: true
  compute_cer: true
  compute_wer: true
  compute_error_breakdown: true
  compute_domain_terms: true
  compute_hallucination_stats: true
```

---

## 20. Agent 작업 순서

Agent는 아래 순서대로 진행한다.

### Step 1. 데이터 확인

- HF Dataset load 가능 여부 확인
- split 확인
- audio/text column 확인
- sample 5개 출력
- transcript normalization 필요 여부 확인

출력:

```text
dataset_report.md
```

### Step 2. baseline 평가

- valid/test에서 baseline Whisper 실행
- CER/WER 산출
- 오류 예시 저장

출력:

```text
outputs/eval_reports/baseline_valid.json
outputs/eval_reports/baseline_test.json
outputs/eval_reports/baseline_examples.md
```

### Step 3. B_H 학습 구현

- Whisper load
- 모든 parameter freeze
- `B_H`만 trainable
- train text만 사용
- CE loss 학습
- checkpoint 저장

출력:

```text
outputs/checkpoints/bias_step_*.pt
```

### Step 4. adapted valid 평가

- checkpoint별 alpha grid 평가
- valid에서 best alpha/checkpoint 선택
- insertion/hallucination 증가 확인

출력:

```text
outputs/eval_reports/adapted_valid_alpha_grid.json
```

### Step 5. 최종 test 평가

- best checkpoint + best alpha만 사용
- test에서 baseline vs adapted 비교
- test를 보고 재튜닝하지 않음

출력:

```text
outputs/eval_reports/final_test_comparison.md
outputs/eval_reports/final_test_predictions.jsonl
```

### Step 6. 결론 작성

결론은 아래 형식으로 작성한다.

```text
- CER/WER 개선 여부
- substitution 개선 여부
- insertion 증가 여부
- domain term recall/precision 변화
- hallucination 증가 여부
- 계속 진행할지 여부
```

---

## 21. 중요한 구현 주의사항

### 21.1 Decoder input / label shift

Whisper 학습에서는 decoder input과 labels shift가 중요하다.

일반적으로:

```text
decoder_input_ids = labels[:, :-1]
target_labels = labels[:, 1:]
```

padding은 `-100`으로 mask한다.

### 21.2 Special token 유지

Whisper special token을 제거하면 안 된다.

언어/task token을 정확히 넣는다.

예:

```text
<|startoftranscript|><|ko|><|transcribe|><|notimestamps|>...
```

### 21.3 Max token length

Whisper decoder max length를 넘는 transcript는 자른다.

너무 긴 transcript는 chunk 단위로 나누는 것이 좋다.

### 21.4 Encoder seq length

Whisper encoder output length는 모델/입력 길이에 따라 달라질 수 있다.

30초 고정 log-mel 입력을 사용하면 일정하게 맞추기 쉽다.

만약 audio 길이에 따라 encoder length가 달라지면:

```text
- B_H를 crop
- B_H를 interpolate
- audio를 30초 chunk로 고정
```

중 하나를 선택한다.

MVP에서는 30초 chunk 고정을 추천한다.

### 21.5 `alpha=1.0` 포함

alpha grid에 반드시 `1.0`을 포함한다.

`alpha=1.0`은 adapted path를 타지만 bias 효과가 없는 baseline sanity check다.

---

## 22. 실패 시 디버깅 순서

### 학습 loss가 안 내려감

확인:

```text
- labels shift가 맞는지
- B_H가 optimizer에 들어갔는지
- B_H requires_grad=True인지
- decoder가 eval mode/freeze 되어도 gradient가 B_H로 흐르는지
- special token이 맞는지
```

### adapted 결과가 baseline과 거의 동일

가능성:

```text
- alpha가 너무 높음
- B_H가 거의 학습되지 않음
- text-only signal이 약함
- decoder가 B_H를 무시함
```

대응:

```text
alpha를 0.8, 0.7로 낮춰보기
learning rate 조정
epochs 증가
B_H init 변경
```

### hallucination 증가

가능성:

```text
- alpha가 너무 낮음
- train text를 과하게 외움
- domain prior가 audio보다 강함
```

대응:

```text
alpha를 0.9 이상으로 올림
epochs 감소
learning rate 감소
B norm regularization 추가
decoder LoRA 금지 유지
```

### domain term은 좋아졌는데 CER/WER 악화

해석:

```text
domain prior가 너무 강해져 일반 단어를 망침
```

대응:

```text
alpha 상향
domain term precision 확인
B regularization 추가
```

---

## 23. 다음 단계 후보

MVP에서 명확한 개선이 있을 때만 다음을 검토한다.

### 2차 후보

```text
- domain-word weighted CE
- B norm regularization
- layer별 B
- decoder 일부 LoRA
```

### 3차 후보

```text
- direct K/V bias
- CTranslate2 runtime 수정
```

### 당분간 보류

```text
- MoE
- routing network
- KL loss
- Bregman-inspired loss
- TTS synthetic audio
```

---

## 24. 최종 판단 기준

MVP 이후 의사결정:

### 계속 진행

```text
test CER/WER 개선
domain term recall 개선
insertion/hallucination 증가 없음
```

### 보류

```text
valid에서는 개선, test에서는 불안정
효과가 특정 alpha에만 민감
짧은 audio에서 hallucination 증가
```

### 중단

```text
baseline보다 악화
insertion 증가
domain term hallucination 증가
효과가 train text memorization으로 보임
```

---

## 25. 요약

이번 MVP는 아래만 한다.

```text
single trainable encoder-output bias B_H
train text only
Whisper 전체 freeze
CE loss only
valid에서 alpha grid
test에서 baseline vs adapted 비교
```

하지 않는다.

```text
MoE
routing
direct K/V bias
CT2 수정
TTS
train audio 사용
decoder full fine-tuning
LoRA
KL/Bregman loss
```

이 실험의 목적은 하나다.

```text
내 도메인 텍스트만으로 Whisper의 substitution/domain-term 오류가 줄어드는가?
그리고 그 과정에서 hallucination/insertion이 늘지 않는가?
```
