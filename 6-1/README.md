# 6-1 실습 — 음성 모델 : 소리를 숫자로 읽고, 정수로 만들고, 다시 소리로

> 현대 AI의 원리와 구조 #6-1 음성 모델 실습 · 뉴런데브클래스
> 전제 : **NVIDIA GPU**(VRAM 16GB 권장 · 8GB 면 01 · 02 · 03 · 05) + 도커가 GPU 를 넘길 수 있는 PC. 4-1 의 Jupyter 이미지(`ai4-lab/jupyter`)를 한 번 빌드했다
> 실습 화면 : Jupyter **http://localhost:8892** → `6-1/notebooks/`
> 모델 · 데이터 : 노트북이 처음 실행될 때 Hugging Face 에서 받아 `hfcache` 볼륨에 둔다(합계 약 25GB). 로그인 · 승인이 필요한 모델은 쓰지 않는다

강의의 자 = **1초를 몇 개의 숫자 · 벡터 · 토큰으로 나타내는가**. 노트북마다 그 숫자를 직접 측정한다.

| 노트북 | 하는 일 | 모델 | 장표 | 시간(RTX 4080 실측) |
|---|---|---|---|---|
| `01_소리를_숫자로.ipynb` | 파형 · STFT · 멜 · 로그를 손으로 → Whisper 입력과 최대 차이 · 초당 몇 개 · 한 파형에 섞인 것 | 없음(계산만) | 음성 모달 문제 · 초당 몇 개 · 로그-멜 입력 | 96초 |
| `02_Whisper_열어보기.ipynb` | 층별 텐서 · 언어 감지 · 손으로 쓴 디코딩 루프 · 교차 어텐션 · 30초 채우기 · 환각 · turbo vs large-v3 | whisper-large-v3-turbo · large-v3 | E2E Transformer · Whisper 구조 · Whisper 텐서 도해 | 119초 |
| `03_ASR_비교와_내데이터.ipynb` | FLEURS 한국어 CER · 지표 정의 · 현장 문장 · 용어 목록 주기 · Qwen3-ASR 안 · 환각 비교 | Whisper turbo · Qwen3-ASR-1.7B · (합성용 Qwen3-TTS) | E2E ASR 세 갈래 · 내 데이터로 측정 | 196초 |
| `04_오디오_LLM.ipynb` | 오디오 토큰 자리표 · 커넥터 · 전사문만 vs 소리(성별) · 질문 방식 · 화자 수 · 길이 비용 | Qwen2.5-Omni-3B | 음성 이해 · 오디오 LLM 텐서 흐름 | 171초 |
| `05_뉴럴_코덱.ipynb` | EnCodec 비트레이트별 · RVQ 손 계산 = 라이브러리 · Mimi 코드북 수별 복원 CER · 10초 = 토큰 몇 개 | EnCodec 24kHz · Mimi · (잣대 Qwen3-ASR) | 뉴럴 코덱 · 코드북 수와 음질 · 오디오 토큰 심화 | 65초 |
| `06_TTS_목소리_만들기.ipynb` | Qwen3-TTS 부품 · 목소리 복제 · 재전사 CER · 화자 유사도 · 참고 길이 · 하이브리드(토큰 → flow matching) 스텝 · CFG · 워터마크 · 내 목소리 | Qwen3-TTS-1.7B · Qwen2.5-Omni-3B · WavLM-SV · AudioSeal | 코덱 토큰 TTS · 2026 코덱 LM · 하이브리드 · 참고 음성 · TTS 측정하는 법 | 457초 |
| `07_실시간_지연.ipynb` | ASR → LLM → TTS 단계별 시간 · 순차 vs 문장 단위 겹치기 · 프레임 예산 | Qwen3-ASR · Qwen3-1.7B · Qwen3-TTS | 실시간 음성 대화 · 지연 예산 | 91초 |

```
6-1/
├─ docker-compose.yml        voice(Jupyter, 8892) 하나
├─ lab/Dockerfile            ai4-lab/jupyter + torch 2.11 · torchaudio 2.11 · transformers 4.57.6 · qwen-asr · qwen-tts · audioseal
├─ notebooks/                01~07 + voice_common.py(데이터 받기 · 들려 주기 · CER). 코드 셀 첫 줄 = "# 셀 N · 제목"
│  ├─ results/               노트북 출력 그림 · 표(.gitignore)
│  └─ _out/                  강사용 실행 사본(.gitignore)
├─ data/                     수강생 녹음(my_voice/) · 노트북이 만든 소리(.gitignore)
├─ tools/                    강사용 : build_notebook_01~07.py(노트북 생성기 — 셀은 여기서 고친다) · nb.py · run_notebooks.sh · show_out.py
│  └─ check_models/          후보 모델 4080 동작 확인(2026-10-09)
└─ 개편안.md                 강의 덱 개편안(진단 · 논증 축 · 신설 장 · 모델 사실표)
```

## 실행 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA, VRAM 16GB 권장. 가장 큰 것 : 04 · 06 의 Qwen2.5-Omni-3B 약 12GB |
| 확인 | `docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L` 가 GPU 를 출력해야 함 |
| 맥 · GPU 없는 PC | 01 만 CPU 로 돈다(나머지는 GPU 필수) |
| 디스크 | 모델 · 데이터 약 25GB(hfcache) + 이미지 7.2GB |

```bash
cd ./-ai-/4-1 && docker compose build          # 처음 한 번 : 바탕 이미지 ai4-lab/jupyter
cd ../6-1
docker compose up -d --build
```

- GPU 고르기 : `VOICE_GPU=1 docker compose up -d` — 번호는 `nvidia-smi -L` 순서. 기본 0
- 바탕화면이 같은 GPU 를 쓰면 VRAM 이 1~2GB 줄어든다. 남은 VRAM 이 모자라면 윈도는 시스템 메모리로 넘겨 **수십 배 느려진다**(오류 없이) — 노트북은 큰 모델을 하나씩 올리고 내린다
- 이미지가 5-1 · 5-2 와 따로인 이유 : torchaudio 의 마지막 판이 2.11 이라 torch 를 2.11 로 맞춤(Qwen3-TTS 가 torchaudio 를 씀), qwen-asr · qwen-tts 가 transformers 4.57.x 를 고정

## 데이터 · 모델

| 무엇 | 출처 · 라이선스 | 크기 |
|---|---|---|
| FLEURS 한국어 test 382문장(270문장 × 여러 화자, 성별 라벨) | `google/fleurs` · CC BY 4.0 | 291MB |
| whisper-large-v3-turbo · large-v3 | OpenAI · MIT | 1.6GB · 3.1GB |
| Qwen3-ASR-1.7B | Qwen · Apache-2.0 · 2026-01 | 4.7GB |
| Qwen3-TTS-12Hz-1.7B-Base | Qwen · Apache-2.0 · 2026-01 | 4.5GB |
| Qwen2.5-Omni-3B | Qwen · Qwen Research License | 11GB |
| Qwen3-1.7B(07 의 답 LLM) | Qwen · Apache-2.0 | 4GB |
| EnCodec 24kHz · Mimi | Meta MIT · Kyutai CC BY 4.0 | 90MB · 380MB |
| WavLM-base-plus-sv · AudioSeal | Microsoft · Meta MIT | 400MB · 수 MB |

- 제외 : Gemma 3n(HF 승인 필요) · Qwen3-Omni(BF16 78GB 이상) · Moshi(PyTorch 24GB) · F5-TTS(의존성이 무거움 — flow matching 은 Qwen2.5-Omni 의 Token2Wav 로 대신)

## 내 목소리 · 내 문장으로

- `data/my_voice/domain_01.wav ~ domain_10.wav` : 03 셀 1 의 현장 문장을 녹음해 넣으면 03 셀 5 가 합성 대신 그 녹음으로 CER 을 측정한다
- `data/my_voice/ref.wav` + `ref.txt` : 06 셀 9 가 그 목소리로 복제(본인 목소리만)
- `data/my_voice/question.wav` : 07 의 질문 소리
- `data/` 는 저장소에 올라가지 않는다(.gitignore)

## 강사용

```bash
cd ./-ai-/6-1/tools
for i in 1 2 3 4 5 6 7; do python build_notebook_0$i.py; done   # 노트북 다시 만들기(출력 없이)
cd .. && VOICE_GPU=1 bash tools/run_notebooks.sh 01 02          # 실행 사본 → notebooks/_out/
python tools/show_out.py 02                                     # 셀별 출력 · 그림 꺼내기
```

## 실측 기록 (2026-10-09, RTX 4080 16GB, 노트북 출력에서 옮김)

| 노트북 · 셀 | 값 |
|---|---|
| 01 셀 3 | 텍스트 토큰 초당 3.3(FLEURS 382문장 평균, Qwen3 토크나이저) · 파형은 그 약 4,900배 |
| 01 셀 4 · 6 | 손 STFT vs torch.stft 최대 상대 차이 2.3e-7 · 손 로그-멜 vs WhisperFeatureExtractor 최대 차이 1.2e-7 |
| 02 셀 3 | [1, 128, 3000] → conv1 [1, 1280, 3000] → conv2 [1, 1280, 1500] → 인코더 출력 [1, 1500, 1280] |
| 02 셀 4 · 5 | 언어 감지 `<|ko|>` 0.9995 · 손 루프 = generate |
| 02 셀 7 | 인코더 시간 1 · 7 · 15 · 30초 입력 모두 62ms |
| 02 셀 8 | 무음 → 「감사합니다.」 · 백색잡음 → 「아멘」 · 모터 소리 → 「시청해주셔서 감사합니다.」 |
| 02 셀 9 | 30문장 CER turbo 0.036 · large-v3 0.034 / 시간 10.6초 · 48.0초 |
| 03 셀 3 · 4 | FLEURS 40문장 CER(구두점 제외) turbo 0.036 · Qwen3-ASR 0.025 |
| 03 셀 5 · 6 | 현장 문장 10개(합성 음성) CER turbo 0.048 · Qwen3-ASR 0.089 → 용어 목록 줌 0.008 · 0.046 |
| 03 셀 7 | Qwen3-ASR 오디오 인코더 12.48초 → 162줄 × 2048(초당 13.0, 설계 12.5) |
| 04 셀 3 · 4 | 오디오 토큰 초당 25.0 · 커넥터 [2048 × 1280] |
| 04 셀 5 | 성별 정답률(20문장) : 소리 + 영어 질문 1.00 · 소리 + 「받아 적고 성별도」 0.90 · 소리 + 「한 단어만, 모르면 모름」 0.05 · 전사문만 0.55 |
| 04 셀 6 | 화자 수 : 한 명 6/6 정답 · 두 명 2/6 정답 |
| 05 셀 3 | EnCodec 복원 CER 1.5kbps 0.132 · 3kbps 이상 0.057(원본과 같음) |
| 05 셀 4 | RVQ 손 계산 = 라이브러리 100% · 남은 오차 48.1 → 5.1 → … → 2.9 |
| 05 셀 5 | Mimi 코드북 1 · 2 · 4 · 8개 복원 CER 0.943 · 0.509 · 0.245 · 0.057 |
| 06 셀 2 | Qwen3-TTS LM 28층 · 폭 2048(1,727M) + 코드 예측기 5층(175M) · 코드북 16개 · 토크나이저 24kHz ÷ 1,920 = 12.5Hz |
| 06 셀 3 · 4 | 화자 3명 × 문장 3개 복제 RTF 1.43~1.54 · 재전사 CER 평균 0.028 → 용어 목록 0.012 |
| 06 셀 5 · 6 | 화자 유사도 복제↔자기 0.959~0.980 · 복제↔다른 화자 0.450~0.662 · 기준 같은 성별 0.924 · 다른 성별 0.475 · 참고 1초 0.410 · 3초 0.984 · 전체 0.978 |
| 06 셀 7 | Talker 토큰 [1 × 605] = 초당 50 · Token2Wav 시각 점 1 → WER 1.00, 2 · 4 · 10 · 32 → 0.053(0.7 · 1.6 · 4.1 · 13.4초) · CFG 0.0001 / 2.0 도 0.053 |
| 06 셀 8 | AudioSeal 탐지 넣기 전 0.0 · 넣은 뒤 1.0 · 8kHz 1.0 · 백색잡음 0.12 · 표시 신호 −28dB |
| 07 셀 3 · 4 · 5 | 받아쓰기 2.11초 · 답 첫 토큰 2.30 · 첫 문장 3.06 · 전체 4.56 · 첫 소리 순차 35.0초 → 문장 단위 15.3초 · TTS 프레임당 243ms(RTF 3.0) |

- 실행 시간 · 지연 값은 실행마다 조금씩 다르다(07 은 LLM 답 길이에 따라 크게 달라짐). 표의 값은 위 날짜의 한 번 실행
- Codex 래퍼 메모 : 로컬 그림을 덱에 넣을 때 그림은 한 번에 한 장(image_uris 에 문자열 하나), 파일은 Codex 작업 폴더 안에 둔다
