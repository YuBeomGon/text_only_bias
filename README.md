# text_only_bias

Whisper **text-only domain adaptation** 실험 (한국어 보험 통화 STT 도메인).
*Domain-Specific Adaptation for ASR through Text-Only Fine-Tuning* 논문 아이디어를 내 데이터에서 검증.

## 개요
음성 없이 **도메인 텍스트만**으로 학습한 bias를 Whisper decoder cross-attention에 주입해
전문용어 오류(substitution / domain-term)가 줄어드는지 검증한다. (데이터·체크포인트는 미포함)

## 실험 진행
| # | 방식 | 결과 |
|---|---|---|
| exp01 | encoder-output bias (가법 혼합) | ❌ 중단 — 전 alpha 악화 |
| exp02 | layer별 K/V bias (A:blend / B:concat) + oracle | ❌ 중단 — 구조적 한계(데이터 아님) 확정 |
| exp03 | **논문 충실 재현** (E_pretrained init + decoder FT + 게이팅 + Bregman, MoE 제외) | 🔧 진행 중 |

→ 전체 결론: [`docs/experiments/CONCLUSIONS.md`](docs/experiments/CONCLUSIONS.md)

## 구조
- `docs/experiments/` — 실험별 설계·체크리스트·결과 (`exp0X/{design,checklist,result}.md`)
- `text_only_bias/common/` — 공통 코드(데이터·메트릭·리포트·모델 유틸)
- `text_only_bias/exp0X_*/` — 실험별 모델·학습·평가
- `text_only_bias/tests/` — TDD 테스트

## 실행 / 테스트
```bash
python -m pytest text_only_bias/tests/ -q
# 사용법: text_only_bias/README.md 참고
```

데이터셋(`hf_dataset`)·체크포인트·예측 산출물은 `.gitignore`로 제외됨.
