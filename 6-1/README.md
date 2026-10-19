# 음성 모델 실습 — 소리를 숫자로 읽고, 정수로 만들고, 다시 소리로

> 현대 AI의 원리와 구조 · 음성 모델 실습 · 뉴런데브클래스
> 전제 : **NVIDIA GPU**(VRAM 16GB 권장 · 파형·멜 실습은 GPU 없이 가능) + 도커가 GPU 를 넘길 수 있는 PC. 환경은 저장소 루트의 `docker-compose.yml` 하나([루트 README](../README.md))
> 실습 화면 : Jupyter **http://localhost:8888** → `6-1/notebooks/` · 커널 **「Python 3 (음성 · 6-1)」**(노트북에 지정돼 있어 열면 자동으로 선택된다)
> 모델 · 데이터 : 노트북이 처음 실행될 때 Hugging Face 에서 받아 `hfcache` 볼륨에 둔다(합계 약 25GB). 로그인 · 승인이 필요한 모델은 쓰지 않는다

강의의 자 = **1초를 몇 개의 숫자 · 벡터 · 토큰으로 나타내는가**. 노트북마다 그 숫자를 직접 측정한다.

| 노트북 | 하는 일 | 모델 | 현재 강의의 연결 |
|---|---|---|---|
| `01_소리를_숫자로.ipynb` | 파형 · STFT · 멜 · 로그를 손으로 → Whisper 입력과 최대 차이 · 초당 몇 개 · 한 파형에 섞인 것 | 없음(계산만) | 파형·FFT·로그멜·초당 개수 |
| `02_Whisper_열어보기.ipynb` | 층별 텐서 · 언어 감지 · 손으로 쓴 디코딩 루프 · 교차 어텐션 · 30초 채우기 · 환각 · turbo vs large-v3 | whisper-large-v3-turbo · large-v3 | 로그멜·인코더·디코더·Whisper 구성도 |
| `03_ASR_비교와_내데이터.ipynb` | FLEURS 한국어 CER · 지표 정의 · 현장 문장 · 용어 목록 주기 · Qwen3-ASR 안 · 환각 비교 | Whisper turbo · Qwen3-ASR-1.7B · (합성용 Qwen3-TTS) | Qwen3-ASR 구조·직접 측정·현장 용어와 환각 |
| `04_오디오_LLM.ipynb` | 오디오 토큰 자리표 · 커넥터 · 전사문만 vs 소리(성별) · 질문 방식 · 화자 수 · 길이 비용 | Qwen2.5-Omni-3B | 음성 이해·커넥터·오디오 LLM 텐서 흐름 |
| `05_뉴럴_코덱.ipynb` | EnCodec 비트레이트별 · RVQ 손 계산 = 라이브러리 · Mimi 코드북 수별 복원 CER · 10초 = 토큰 몇 개 | EnCodec 24kHz · Mimi · (잣대 Qwen3-ASR) | 코드북·뉴럴 코덱·음향/의미 토큰·VALL-E 분업 |
| `06_TTS_목소리_만들기.ipynb` | Qwen3-TTS 부품 · 목소리 복제 · 재전사 CER · 화자 유사도 · 참고 길이 · 하이브리드(토큰 → flow matching) 시각 점 수 · CFG · 워터마크 · 내 목소리 | Qwen3-TTS-1.7B · Qwen2.5-Omni-3B · WavLM-SV · AudioSeal | 프레임 안 분업·Flow Matching·참고 음성·TTS 측정·워터마크 |
| `07_실시간_지연.ipynb` | ASR → LLM → TTS 단계별 시간 · 순차 vs 문장 단위 시작(측정값 기반 계산) · 프레임 예산 | Qwen3-ASR · Qwen3-1.7B · Qwen3-TTS | 파이프라인·첫 소리까지 지연·지연 도해 |

```
6-1/
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
| 확인 | 저장소 루트에서 `docker compose exec jupyter nvidia-smi -L` 이 `.env` 의 `LAB_GPU` 로 정한 GPU 하나를 출력해야 함 |
| 맥 · GPU 없는 PC | 01 만 CPU 로 돈다(나머지는 GPU 필수) |
| 디스크 | 모델 · 데이터 약 25GB(hfcache) + 공통 Jupyter 이미지 |

```bash
cd ./-ai-
cp env.example .env              # 처음 한 번 (LAB_GPU = 쓸 GPU 번호)
docker compose up -d --build      # 모든 강의 공통 Jupyter(ai4-jupyter) + db
```

- GPU 지정 : 루트 `.env` 의 `LAB_GPU` — 번호는 `nvidia-smi -L` 순서. 기본 0. 바꾼 뒤 `docker compose up -d`
- 바탕화면이 같은 GPU 를 쓰면 VRAM 이 1~2GB 줄어든다. 남은 VRAM 이 모자라면 윈도는 시스템 메모리로 넘겨 **수십 배 느려진다**(오류 없이) — 노트북은 큰 모델을 하나씩 올리고 내린다
- 6-1 만 커널이 따로(venv `/opt/voice`, 커널 이름 `voice`)인 이유 : torchaudio 의 마지막 판이 2.11 이라 torch 를 2.11 로 맞춤(Qwen3-TTS 가 torchaudio 를 씀), qwen-asr · qwen-tts 가 transformers 4.57.x 를 고정

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
for i in 1 2 3 4 5 6 7; do python build_notebook_0$i.py; done   # 노트북 다시 만들기(출력 없이, 커널 voice 지정)
cd .. && bash tools/run_notebooks.sh 01 02                      # 떠 있는 ai4-jupyter 안에서 voice 커널로 실행 → notebooks/_out/
python tools/show_out.py 02                                     # 셀별 출력 · 그림 꺼내기
```

## 기존 실측 기록 (주로 2026-10-09, RTX 4080 16GB)

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

- 기존 표와 실습 덱의 수치는 과거 실행 예시. 편집 전 실행 사본은 `slides/work/baseline_out/`에 보존. 이번 재실행 사본은 `notebooks/_out/`에 저장하며 값이 달라질 수 있음
- 실행 시간·지연 값은 실행마다 다름. 지연 실습은 LLM 답 길이와 세 모델의 동시 상주 조건에 영향받음
- Codex 래퍼 메모 : 로컬 그림을 덱에 넣을 때 그림은 한 번에 한 장(image_uris 에 문자열 하나), 파일은 Codex 작업 폴더 안에 둔다

## 현재 강의와 실습의 연결 범위

- 코드북 수별 CER은 뉴럴 코덱·음향/의미 토큰·VALL-E 분업의 판단 근거. 입력 한 문장의 결과이며 음질 지표가 아님
- Token2Wav의 `num_steps`는 시각 점 수. RK4(Runge–Kutta 4차 적분)는 구간당 방향 예측 4회, CFG(Classifier-Free Guidance, 조건 유도)를 사용하면 조건·무조건 예측을 배치로 묶어 실행. DiT 호출 수 = 4 × (점 수 − 1), 배치 내 예측 개수는 그 2배. 점 1개는 적분 구간 0개. CFG 0.0001은 효과가 작지만 계산 비용은 기본값과 같음
- 화자 유사도는 동일인 정답률이 아님. 목소리 복제 셀 5에서 전체 녹음 쌍의 값과 오디오 LLM에 사용한 남·여 6쌍을 CSV로 저장·출력. 성별 라벨만으로 같은 성별의 녹음을 다른 사람이라고 판정할 수 없음. 강의의 별도 화자 실험 수치는 이 실습의 재현값으로 사용하지 않음
- 지연 실습의 35.0초·15.3초는 단계별 측정값을 더한 첫 재생 가능 시각. 실제 겹침 실행이나 첫 오디오 프레임 도착을 측정한 값이 아님. 프레임당 243ms는 전체 TTS 시간에서 계산한 평균
- TTS 단독 RTF 1.43~1.54와 세 모델 동시 상주 RTF 3.04는 입력·배치·참고 처리·메모리 조건이 다름. 기존 출력만으로 어느 요인이 차이를 만들었는지는 확인할 수 없음
- 실행 모델: Whisper·Qwen3-ASR·Qwen3-TTS·Qwen2.5-Omni·Qwen3 텍스트 LLM·EnCodec·Mimi·WavLM·AudioSeal. Nemotron-3.5-ASR·Voxtral Realtime은 이 실습에서 실행하지 않음
- 설명만 다루는 범위: FFT 필터 적용, ASR 스트리밍·학습, VAD, 화자 분리, AudioLM·X-Codec2, k-means 학습·IVF, Qwen3-Omni Thinker/Talker 실행, Moshi 전이중, 보고서의 초당 6.25토큰. 실습의 부품 확인과 해당 모델의 실행 검증을 구분

## RESULT — 실습 덱 수정 (2026-10-13)

- 덱 ID: `15mop0IIdh7WWhM_NT6Wg8S0szUVCqtJA3bVL-XhqDrE` · 28장 유지
- revisionId: `qYhQIzpXnglYoA`
- 변경 문구 재읽기·첫 줄/끝 줄 대조 및 최종 28장 썸네일 시각 확인. 본문 폭·문단 간격 조정, 로고·회색 블록·그림 겹침과 잘림 확인
- 강의 덱은 편집 대상 밖. 강의의 걸음 정의·실습하지 않는 모델 표기·별도 화자 실험 수치의 문제는 남아 있음

| 슬라이드 ID | 객체 ID | 변경 |
|---|---|---|
| `g3ef1317cf86_1_0` | `g3ef1317cf86_1_3` | 강의 번호 대신 음성모델 실습 표기 |
| `p61_env` | `p61_env_t`, `p61_env_b` | 환경 제목·GPU 조건·내용 중심 표현 |
| `p61_map` | `p61_map_t`, `p61_map_b` | 실행 모델·문헌 설명·미실행 범위 구분 |
| `p61_02c` | `p61_02c_t`, `p61_02c_b` | 환각 대책의 미실행 표시·학습 손실 셀 연결 |
| `p61_03a` | `p61_03a_b` | 기존 실행 출력의 시간 37초·50초와 로딩 포함 조건 |
| `p61_03b` | `p61_03b_b` | 강의 번호 제거·현장 입력과 표기 기준 |
| `p61_04a` | `p61_04a_b` | 이미지 자리표와 연결·Qwen2.5 부품 확인 범위 |
| `p61_04b` | `p61_04b_b` | 화자 수 답변과 벡터 유사도 구분 |
| `p61_05a` | `p61_05a_b` | 잔차 크기·CER·음질 구분, 실측 유지 |
| `p61_05b` | `p61_05b_b` | 의미 학습과 단독 파형 복원 구분, 실측 유지 |
| `p61_06b` | `p61_06b_b` | 동일인 판정·청감·한 화자 참고 길이 결과의 범위·이번 실행 남/여 6쌍 0.419~0.542 |
| `p61_06c` | `p61_06c_t`, `p61_06c_b` | 시각 점·적분 구간·호출과 배치 내 예측 구분·CFG |
| `p61_07a` | `p61_07a_t`, `p61_07a_b`, `p61_07a_img` | 측정과 계산·첫 텍스트 조각·평균 프레임 비용·계산 도해 |

- 본문 상자 19개(`*_b`)의 폭·문단 간격, 일반 제목 상자(`*_t`)의 폭 조정. 슬라이드 삭제·추가·순서 변경 없음
- 스냅샷: `slides/work/final_01.png`~`final_28.png` · 변경 기록: `slides/work/slides_result.json`

## 수정 후 GPU 실행 확인 (2026-10-13)

- RTX 4080 · PyTorch `2.11.0+cu130` · transformers `4.57.6`. Windows Docker에서 `run_notebooks.sh`와 같은 이미지·마운트·GPU 환경변수로 7개 노트북을 순차 실행
- 7개 모두 Docker 실행 종료 코드 0. 실행 사본의 모든 코드 셀에 실행 번호가 있고 오류 출력 없음. 실행 코드가 생성기에서 만든 최종 노트북 코드와 일치
- 파형·멜: 최대 차이 `1.1920928955078125e-07`. Whisper 학습: 손 계산·라이브러리 손실 `0.414002`, 인코더 conv1까지 기울기 출력
- 코드북 수별 CER과 화자 성별·화자 수 실험은 기존 예시와 같은 값. 추가 남/여 6쌍 유사도 `0.462, 0.445, 0.419, 0.427, 0.526, 0.542`
- Token2Wav: 점 1개 WER `1.000`, 점 2·3·4·10·32개 `0.053`; 점 10개 CFG `0.0001/0.5/2.0` 모두 `0.053`. 점 2개 호출 4회·배치 내 예측 8개. 횟수 열은 설치된 모델 소스의 RK4·배치 구조로 계산한 값
- 이번 TTS 단독 RTF `1.59~1.77`. 세 모델 동시 상주 지연 실습 RTF `3.35`, 평균 프레임당 `268ms`. 첫 재생 가능 시각은 순차 `32.86초`, 문장 단위 `20.09초`로 계산. 실제 겹침 실행 결과가 아님
- AudioSeal 잡음 탐지 `0.286`으로 과거 `0.119`와 다름. 생성 음성과 잡음이 실행마다 달라질 수 있으므로 과거 값을 일반 성능으로 취급하지 않음
- 사용자 녹음 `ref.wav/ref.txt`가 없어 사용자 목소리 복제 분기는 건너뜀. 공개 FLEURS 입력의 복제 실험은 실행
- 출력에 Flash Attention 미설치 경고가 있음(수동 PyTorch 경로 사용). 실행 로그에 Jupyter TCP 통신 경고가 있음. WSL은 Docker 소켓 연결 실패로 실행하지 못해 Windows Docker로 전환. 기존 실행 스크립트는 실패를 숨길 수 있어 종료 코드만으로 판정하지 않고 실행 사본을 직접 확인
- 실행 로그: `slides/work/gpu-native.log` · 셀별 출력·오류·코드 일치 기록: `notebooks/results/alignment_verification_2026-10-13.json` · 쌍별 유사도: `notebooks/results/06_speaker_pairs.csv`
