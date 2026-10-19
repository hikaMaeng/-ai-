# 현대 AI의 원리와 구조 — 실습 저장소

뉴런데브클래스 「현대 AI의 원리와 구조」 강의의 실습 자료입니다.
모든 실습은 **Docker Desktop** 과 **LM Studio**(임베딩 서버)만 있으면 따라 할 수 있습니다. 실습 화면은 Jupyter 노트북입니다.

| 강 | 주제 | 폴더 |
|---|---|---|
| 3강 (3-1) | 임베딩 벡터 연산 — 평균(중심 벡터 추천) · 가감(속성 바꾸기) · 프로토타입 분류 · Rocchio 피드백을 산업별 공개 데이터 10종으로 (Jupyter · 실습마다 따로 적재 · 임베딩) | [3-1](3-1/) |
| 4강 (4-1) | 벡터디비 — 데이터 적재·임베딩(LM Studio) · 한국어 전문검색 · BM25 직접 구현 · pgvector 벡터 검색 · 필터 (Jupyter) | [4-1](4-1/) |
| 4강 (4-2) | 벡터디비 — HNSW · IVF · PQ 직접 구현 · pgvector 인덱스 측정 · 점수 퓨전 · 분포 진단 (Jupyter) | [4-2](4-2/) |
| 5강 (5-1) | 이미지 생성 — CLIP 구동 : 이미지·텍스트 타워 텐서 · N×N 채점표 · 제로샷 분류 · 검색 · 관계/부정/개수 한계 · 멀티모달 임베딩(Qwen3-VL-Embedding-2B) : 지시문 · 한국어 · 섞인 입력 · 문서 스크린샷 검색 · MRL · 사진 검색 · VLM(Qwen3-VL-2B) : 자리표 · 텐서 흐름 · 커넥터 · 문서 판독 · 부정/관계/개수 · 환각 · 리랭커 · 설명문 색인 (Jupyter) | [5-1](5-1/) |

## 실습 환경 — compose 하나 · Jupyter 컨테이너 하나

모든 강의가 저장소 루트의 `docker-compose.yml` 하나와 Jupyter 컨테이너 하나(`ai4-jupyter`)를 같이 쓴다. 강의 폴더에는 compose 파일이 없다.

```
docker-compose.yml     jupyter(8888) · db(5432) + 선택 pgadmin(5050) · comfy(8188)
env.example            GPU 번호 · LM Studio 주소 설명 → .env 로 복사해서 쓴다
docker/lab/            Jupyter 이미지(ai4-lab/jupyter-all) — 파이썬 커널 둘
docker/db/             pgvector(PostgreSQL 17) + 한국어 형태소 분석기 · 최초 스키마(init/)
docker/pgadmin/        pgAdmin 서버 등록 정보
```

### 1. 준비 · 기동

```bash
cd ./-ai-
cp env.example .env              # 처음 한 번. GPU 번호 · LM Studio 주소를 내 PC 에 맞게 고친다
docker compose up -d --build      # jupyter + db. 첫 빌드는 PyTorch 두 벌을 받아 10~20분
```

- 강사 PC 실측(2026-10-19) : 캐시 없는 첫 빌드 18분 16초(이미지 13.6GB). 음성 커널에 패키지를 더한 재빌드는 캐시를 써서 6분 14초, 최종 이미지 13.8GB(PyTorch CUDA 판 두 벌 포함)

- 실습 화면 : **http://localhost:8888** (로그인 없음) → 왼쪽 파일 목록에서 강의 폴더 → `notebooks`
- 멈추기 : `docker compose down` (컨테이너만 지운다. 모델 · 데이터셋 · DB 는 볼륨에 남는다)
- 전부 처음부터 : `docker compose down -v` — **볼륨까지 지운다**(받아 둔 모델 · 데이터셋 · DB 임베딩이 사라진다)
- `.env` 를 고친 뒤에는 `docker compose up -d` 로 컨테이너를 다시 만들어야 적용된다(실행 중인 노트북은 끊긴다)

### 2. 커널 선택

노트북을 열면 커널이 노트북에 적힌 대로 정해진다. 오른쪽 위 커널 이름을 눌러 바꿀 수 있다.

| 커널 | 이름 | 쓰는 강의 | 주요 패키지 |
|---|---|---|---|
| Python 3 (기본) | `python3` | 3-1 · 4-1 · 4-2 · 5-1 · 5-2 · 6-2 · 8-1 | torch 2.14(CUDA 13.0) · transformers 5.17 · diffusers 0.40 · psycopg · pgvector |
| Python 3 (음성 · 6-1) | `voice` | 6-1 | torch 2.11 · torchaudio 2.11 · transformers 4.57.6 · qwen-asr · qwen-tts |

6-1 이 따로인 까닭 : torchaudio 는 2.11 이 마지막 판이고 qwen-asr · qwen-tts 가 transformers 4.57.x 를 고정한다.

### 3. GPU 지정

`.env` 의 `LAB_GPU` 에 `nvidia-smi -L` 의 번호를 준다(한 장이면 0). 컨테이너 안에서는 그 GPU 하나만 보인다(`cuda:0`).

```bash
nvidia-smi -L                     # 번호 확인
docker compose exec jupyter python -c "import torch; print(torch.cuda.get_device_name(0))"
```

- GPU 가 없는 PC : `docker-compose.yml` 의 `jupyter` 아래 `deploy` 블록을 지우고 올린다(GPU 가 필요 없는 강의만 된다)
- 도커가 GPU 를 넘기는지 확인 : Windows 는 Docker Desktop(WSL2) + NVIDIA 드라이버, Linux 는 NVIDIA Container Toolkit

### 4. 실습 기준 PC · 자원 상한

실습은 **수강생 PC** 를 기준으로 만든다 — 노트북 i7 · RAM 16GB · NVIDIA VRAM 8GB(또는 통합 메모리 16GB 맥).

| 자원 | 기준 | 어떻게 지키나 |
|---|---|---|
| CPU | 노트북 i7 ≈ 데스크톱 코어 6개 | compose 상한 : Jupyter `LAB_CPUS=6` · DB `DB_CPUS=2` |
| 메모리 | RAM 16GB 중 도커 12GB | compose 상한 : Jupyter `LAB_MEM=9g` · DB `DB_MEM=3g`(스왑 없음 → 넘으면 커널이 죽는다) |
| VRAM | 8GB 목표 · **12GB 상한**(넘으면 실습 부적합, 넘는 부분은 오프로딩 · CPU) | 도커로 제한 불가 → 노트북마다 실행 기록에서 측정(이슈 #3) |

- 강사 PC 처럼 큰 머신도 같은 상한으로 돌려 수강생 PC 에서 되는지 확인한다. 값은 `.env` 로 바꾼다(`env.example` 참고)
- 맥의 도커에는 NVIDIA GPU 가 없다 — GPU 실습은 CPU 로 돌며 크게 느려진다

### 5. 선택 서비스(profile)

| 서비스 | 켜기 | 화면 | 쓰는 곳 |
|---|---|---|---|
| pgAdmin | `docker compose --profile pgadmin up -d` | http://localhost:5050 | 3-1 · 4-1 · 4-2 DB 를 눈으로 확인 |
| ComfyUI | `docker compose --profile comfy up -d` | http://localhost:8188 | 5-2 의 03 · 05 (모델은 `5-2/tools/download_models.sh`) |

- ComfyUI 와 Jupyter 를 서로 다른 GPU 에 두려면 `.env` 의 `COMFY_GPU` 와 `LAB_GPU` 를 다르게 준다
- 노트북 안에서 ComfyUI 주소는 `http://comfy:8188`(환경변수 `COMFY_URL`)

### 6. LM Studio(3-1 · 4-1 · 4-2 임베딩 · 8-1 LLM)

LM Studio 는 컨테이너 밖, 내 PC 에서 돈다. 컨테이너 안의 `localhost` 는 컨테이너 자신이라 노트북은 `host.docker.internal` 로 닿는다.
8-1 은 `.env` 의 `LMSTUDIO_URL` · `AGENT_MODEL` · `AGENT_MODE` · `AGENT_MIN_TOKENS` 를 컨테이너 환경변수로 받는다(설명은 `env.example`).
3-1 · 4-1 · 4-2 노트북은 `.env` 를 읽지 않고 **셀 1 의 `LMSTUDIO_URL`**(기본 `http://host.docker.internal:1234/v1`)을 쓴다. LM Studio 포트가 1234 가 아니면 그 셀의 포트를 바꾼다.

### 7. 강사용 일괄 실행

`6-1` · `6-2` · `8-1` 의 `tools/run_notebooks.sh` 는 떠 있는 `ai4-jupyter` 안에서 `docker compose exec` 로 nbconvert 를 돌린다 — 새 컨테이너를 만들지 않는다. 결과 사본은 각 `notebooks/_out/`.

```bash
bash 6-1/tools/run_notebooks.sh 01     # 음성 커널(voice)로 실행
bash 6-2/tools/run_notebooks.sh        # 기본 커널
```
