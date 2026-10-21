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
docker/lab/            Jupyter 이미지(ai4-lab/jupyter-all) — 파이썬 커널 둘 · cuda_init_check.py(CUDA 첫 초기화 진단, 8절)
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
| VRAM | 8GB 목표 · 12GB 가이드(넘는 부분은 오프로딩 · CPU) | 도커로 제한 불가 → 노트북마다 실행 기록에서 측정(이슈 #3) |

- 강사 PC 처럼 큰 머신도 같은 상한으로 돌려 수강생 PC 에서 되는지 확인한다. 값은 `.env` 로 바꾼다(`env.example` 참고)
- 가이드를 넘는 실습도 잘못된 것은 아니다 — 그 내용을 실현하려면 그만한 자원이 든다. 그런 노트북은 첫 셀에 **「필요 자원(실측)」** 을 적어 둔다(수강생은 실행 전에 자기 PC 와 비교)
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

### 8. CUDA 첫 초기화가 느릴 때(Windows · WSL2)

증상 : 새 프로세스(노트북 커널 · `docker compose exec`)마다 첫 `torch.cuda.is_available()` 이 3분 반가량 걸린다. 같은 프로세스 안의 두 번째 호출부터는 0초. CPU 만 쓰는 모델도 첫 `backward()` 에서 같은 시간을 기다린다(역전파 엔진이 시작할 때 CUDA 장치 수를 묻기 때문).

먼저 어느 층에서 시간을 쓰는지 가른다. 단계마다 새 프로세스를 띄워 각각을 「첫 호출」로 측정한다.

```bash
docker compose exec jupyter python /work/docker/lab/cuda_init_check.py
```

| 단계 | 정상일 때(강사 PC, 2026-10-21 07:28 UTC) | 느리면 |
|---|---|---|
| 드라이버 파일 읽기(9p) | 163 MB · 1.32~1.40 s | Windows DriverStore 공유(9p)가 느림 |
| `nvidia-smi -L` | 0.06~0.07 s | NVML 경로 |
| `cuInit`(ctypes) | 0.10~0.11 s | 드라이버 · WSL2 GPU 통로(`/dev/dxg`) |
| `cudaGetDeviceCount`(libcudart 13) | 0.10 s | CUDA 런타임 |
| `torch.cuda.is_available()` | 0.11 s(`import torch` 1.7~1.8 s) | torch 쪽(위 단계가 모두 빠를 때만) |

`torch.cuda.is_available()` 은 `torch._C._cuda_getDeviceCount()` → `cudaGetDeviceCount` → `cuInit` 을 그대로 부른다. 그래서 위 층이 빠른데 torch 만 느린 경우가 아니면 이미지 · compose 를 고쳐도 소용없다. WSL2 에서 `libcuda.so.1.1` · `nvcubins.bin` · `libnvidia-ptxjitcompiler.so.1` 은 이미지 안이 아니라 `/usr/lib/wsl/drivers/` 에서 읽힌다. 이 경로는 Windows 드라이버를 9p 로 공유한 것이다.

**강사 PC 기록(RTX 4080 `LAB_GPU=1` · 드라이버 617.14 · Docker Desktop WSL2 커널 6.6.87.2)**

| 시각(UTC) | 측정 | 결과 |
|---|---|---|
| 10-20 03:03 | `docker compose start jupyter`(호스트 재부팅 뒤 첫 시작) | — |
| 10-20 03:10 ~ 03:21 | 작은 학습 스크립트 · nbconvert 의 첫 학습 | 216.8 s · 214.9 s(학습 자체는 1.2~1.4 s) |
| 10-20 03:21 ~ 03:28 | CPU 모델 첫 `backward()`, `CUDA_VISIBLE_DEVICES=` · 스레드 1 / 6 | 209.854 s / 209.850 s |
| 10-20 03:29 ~ 03:36 | GPU 보임, 첫 `backward()` / `is_available()` | 211.554 s / 210.2 s |
| 10-21 03:53 | 임베딩 모델 시험 스크립트의 `is_available()` | 210.3 s |
| 10-21 04:00 ~ 04:31 | 같은 종류 스크립트(명령 전체 소요 275~417 s, 단계별 시간 없음) | 느렸던 것으로 추정되는 마지막 구간 |
| 10-21 07:19 ~ 07:28 | 같은 명령 · `cuda_init_check.py` · 첫 `backward()` | `is_available()` 0.2~0.3 s · 첫 `backward()` 0.123~0.181 s |

- 컨테이너는 10-20 03:03 부터 10-21 07:24 까지 재시작하지 않았다. 그런데 07:19 에는 이미 빨랐다 → 컨테이너 · 이미지를 건드리지 않았는데 저절로 풀렸다. 원인은 호스트(드라이버 · WSL2 GPU 통로) 쪽의 일시적인 상태로 본다. 느린 상태에서 `cuInit` 을 따로 측정하지 못해서 어느 층인지는 **확정하지 못했다**
- 정상인 지금 같은 조건을 만들어 재현을 시도했으나 모두 0.2~0.3 s 였다(가설 기각) :
  - 컨테이너 메모리 압박(익명 메모리 8.1 GB / 상한 9 GB)
  - CPU 상한 포화(12 프로세스, 608 %)
  - 같은 GPU 를 다른 프로세스가 100 % 사용(8192² 행렬곱 반복). 컨텍스트 생성 0.4 s
  - 보이지 않는 3090 이 VRAM 21.5 GB · 사용률 48 % 인 상태
  - `docker compose restart jupyter` 직후
- 같은 시간대 Windows 시스템 이벤트에는 GPU 드라이버 재시작(TDR) · 장치 변경 기록이 없었다. Defender 정밀 검사도 없었다(빠른 검사 10-21 00:30~00:34 KST 뿐)

회피책
- 느린 상태에서는 **프로세스마다 한 번** 기다린다. 커널을 살려 두면 그 안의 셀은 다시 기다리지 않는다. 강의 전에 커널을 켜고 `import torch; torch.cuda.is_available()` 을 한 번 돌려 둔다. nbconvert 일괄 실행은 노트북마다 새 커널이라 노트북마다 기다린다
- 재발하면 먼저 `cuda_init_check.py` 를 **처음 한 번** 돌려 느린 층을 기록한다(9p 읽기는 두 번째부터 캐시가 섞인다). 그 결과를 이 표에 더한다
- 풀리지 않으면 Docker Desktop 재시작 → 그래도 안 되면 `wsl --shutdown` 뒤 Docker Desktop 재시작 순서로 해 본다. **이 조치가 효과 있는지는 아직 확인하지 못했다**(이번에는 조치 전에 저절로 풀렸다)
