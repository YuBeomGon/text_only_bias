# 실험 인덱스

Whisper text-only domain adaptation 실험 기록. 공통 배경은 상위
[whisper_text_only_bias_mvp_guide.md](../whisper_text_only_bias_mvp_guide.md)와
[dataset_description.md](../dataset_description.md) 참고.

> **▶ 전체 종합 결론: [CONCLUSIONS.md](./CONCLUSIONS.md)** (필독)

| # | 실험 | 방식 | 상태 | 결과 |
|---|---|---|---|---|
| exp01 | [encoder-output bias](./exp01_encoder_output_bias/design.md) | 단일 전역 `B_H` 가법 혼합 (=실은 **논문 아키텍처**) | ✅ 완료 | ❌ **중단** — 전 alpha 악화 ([result](./exp01_encoder_output_bias/result.md)) |
| exp02 | [direct K/V bias](./exp02_kv_bias/design.md) | layer별 K/V (A:blend / B:concat) + oracle probe | ✅ 완료 | ❌ **중단** — A·B·oracle 모두 미개선 ([result](./exp02_kv_bias/result.md)) |
| exp03 | [논문 충실 재현](./exp03_paper_faithful/design.md) | **E_pretrained 초기화(#1) + decoder fine-tune(#3)** + 게이팅·Bregman 구현, kv-cache 레벨 주입 | ✅ 완료 | ❌ **중단** — P2 init 단독 무효, P3 decoder FT 파국 ([result](./exp03_paper_faithful/result.md)) |

각 실험 폴더: `design.md`(설계/플랜) · `checklist.md`(진행) · `result.md`(결과/판정).

## 핵심 교훈
1. 단일 글로벌 text-only bias(B만 학습)는 **구조적으로** baseline 개선 불가 — 오라클(정답 텍스트 학습)도
   무변화로 입증. **데이터 증량은 이 방식엔 답 아님.**
2. **사후 논문 정독**: 논문의 "K/V bias"는 encoder-output 대체(=exp01)이며, 우리는 **①B 초기화=E_pretrained,
   ②decoder fine-tune**를 놓침 → exp03에서 보완. 자세히는 CONCLUSIONS.md §4.
3. **exp03 결과**: E_pretrained init를 해도 init 단독(P2)은 동일 붕괴, decoder full FT(P3)는 **더 악화**
   (train 때 K/V=B로 학습 → audio 추론과 불일치 + 과적합 → CER 5+·환각). **논문 재현으로도 이 데이터에선 실패.**
4. 모든 실험은 **baseline sanity(bias off == plain Whisper) 토큰단위 일치**를 먼저 통과 (전부 통과).
5. 다음 후보(exp04): train/infer 조건 일치(학습 시 audio K/V 혼합) · LoRA 경량 FT · **shallow fusion**(per-token 주소·환각 통제 용이).
