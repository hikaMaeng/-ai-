# 5강 실습 — CLIP : 사진과 글을 한 공간에서 재기

> 현대 AI의 원리와 구조 #5-1 이미지 생성 실습 · 뉴런데브클래스
> 전제 : [4-1](../4-1/) 의 도커 환경을 한 번 띄운 적이 있다(`ai4-lab/jupyter` 이미지 · `hfcache` 볼륨). **LM Studio 는 필요 없다.**
> 실습 화면 : 4-1 의 Jupyter **http://localhost:8888** 에서 `5-1/notebooks/01_CLIP_실습.ipynb` 를 열면 된다 — 셀 2 가 처음 한 번 PyTorch 를 설치(CPU 판 약 200MB, 1~3분)
> 장치 : 노트북 셀 1 의 `DEVICE` 로 NVIDIA GPU(`cuda`) · 맥 Metal(`mps`) · `cpu` 를 고른다. 기본 `auto`

장표 「CLIP 대조학습」 · 「CLIP의 활용과 한계」 · 「이미지·텍스트 임베딩 파이프라인」(이미지 타워 · 텍스트 타워 · 정렬·손실)을
CLIP ViT-B/32 로 직접 돌려, 장표의 텐서 모양과 주장(제로샷 분류 · 검색 · 관계/부정/개수의 한계)을 숫자로 확인한다.

```
5-1/
├─ docker-compose.yml     4-1 스택 옆에 CLIP 용 Jupyter(ai4-clip, 포트 8889) 하나를 더 띄운다 (CPU)
├─ docker-compose.gpu.yml 위 파일에 덧붙이면 같은 컨테이너를 CUDA 판 PyTorch + GPU 로 띄운다
├─ clip/Dockerfile        ai4-lab/jupyter + PyTorch(CPU 또는 CUDA) + transformers
├─ notebooks/
│  └─ 01_CLIP_실습.ipynb   셀 13개. 코드 셀 첫 줄 = "# 셀 N · 제목", 설정은 셀 1
├─ images/                실습 사진 150장 + images.csv(파일 · 그룹 · 라벨 · 출처 · 라이선스)
└─ tools/                 강사용 — 수강생은 실행하지 않는다
   ├─ prepare_images.py   images/ 를 다시 만드는 스크립트(원본 데이터셋에서 선별)
   └─ build_notebook.py   노트북 생성기(셀 내용은 여기서 고친다)
```

## 실행 환경 — 다섯 중 하나

| 경로 | 셀 1 `DEVICE` | 누가 | 방법 |
|---|---|---|---|
| ⓪ 4-1 Jupyter 그대로 | `cpu` | 누구나(기본) · 준비 없음 | http://localhost:8888 에서 노트북을 열고 Run All. 셀 2 가 PyTorch 를 설치 |
| ① 도커 CPU | `cpu` | 4-1 컨테이너를 건드리지 않고 따로 띄우고 싶을 때 | `docker compose up -d --build` → http://localhost:8889 |
| ② 도커 GPU | `cuda` | NVIDIA GPU + 도커가 GPU 를 넘길 수 있는 PC | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build` → 8889 |
| ③ 직접 설치 · NVIDIA | `cuda` | 도커 GPU 가 안 되는 윈도 · 리눅스 PC | 아래 '직접 설치' |
| ④ 직접 설치 · 맥 | `mps` | Apple Silicon 맥(Metal) | 아래 '직접 설치' |

- ⓪ : 셀 2 가 `torch` · `transformers` 가 없으면 그 Jupyter 에 설치한다(`pip install`). 4-1 컨테이너에는 GPU 가 연결되어 있지 않으므로 CPU 판
  - 설치는 컨테이너 안에 남는다. `docker compose up -d --build` 등으로 4-1 컨테이너를 다시 만들면 셀 2 가 다시 설치한다
  - 받는 판 : 리눅스 · 윈도에서 NVIDIA GPU(`nvidia-smi`)가 보이면 CUDA 판, 맥은 기본판(Metal), 그 밖은 CPU 판 — ③ · ④ 에서 `pip install` 을 건너뛰어도 셀 2 가 같은 판을 받는다
- `DEVICE = "auto"` 는 `cuda` → `mps` → `cpu` 순으로 있는 것을 고른다. 셀 2 가 고른 장치 이름을 출력한다
- 없는 장치를 고르면 셀 2 가 "쓸 수 있는 장치" 목록과 함께 멈춘다
- 맥의 Metal 은 도커 컨테이너 안에서 쓸 수 없다(도커가 리눅스 가상머신에서 돈다) → 맥은 ④
- 장치를 바꿔도 결과 숫자는 같다(아래 실측). GPU 에서 CPU 와 같은 숫자가 나오도록 셀 2 가 TF32(저정밀 곱셈)를 끈다

### ① · ② 도커

```bash
cd ./-ai-/5-1
docker compose up -d --build                                                  # ① CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build  # ② GPU
```

- 4-1 의 `ai4-lab/jupyter` 이미지 위에 PyTorch(`torch==2.14.*`) · `transformers==5.17.*` 만 더한다. 4-1 · 4-2 이미지는 그대로다
- ① 이미지 `ai4-lab/jupyter-clip` 1.74GB(CPU 휠) · ② `ai4-lab/jupyter-clip:gpu` 6.48GB(CUDA 13.0 판). 첫 빌드는 PyTorch 를 받느라 수 분
- 두 경로 모두 같은 컨테이너 이름 `ai4-clip` · 같은 포트 8889 — 명령을 바꿔 치면 서로 바뀐다
- ② 전제 : 아래가 GPU 이름을 출력해야 한다. 실패하면 ③ 으로
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L
  ```
- 컨테이너 `ai4-clip` 은 4-1 과 같은 compose 프로젝트(`ai4-lab`)에 들어가고 `hfcache` 볼륨을 같이 쓴다 → CLIP 모델(약 600MB)을 한 번 받으면 남는다
- 포트 8889 는 내 PC(127.0.0.1)에서만 열린다. 겹치면 `CLIP_PORT=8890 docker compose up -d`

### ③ · ④ 직접 설치 (도커 없이)

Python 3.12 이상. 저장소의 `5-1` 폴더에서 :

```bash
python -m venv .venv
source .venv/bin/activate            # 윈도 : .venv\Scripts\activate
# PyTorch — 하나만
pip install torch==2.14.* --index-url https://download.pytorch.org/whl/cu130   # ③ NVIDIA (윈도 · 리눅스, 드라이버 580 이상)
pip install torch==2.14.*                                                       # ④ 맥 (Metal 포함)
# 나머지
pip install "transformers==5.17.*" safetensors pillow pandas matplotlib jupyterlab
jupyter lab notebooks/01_CLIP_실습.ipynb
```

- 사진 폴더는 셀 1 `IMAGE_DIR = "auto"` 가 노트북 옆의 `../images` 로 찾는다
- 모델은 `~/.cache/huggingface` 에 받는다(약 600MB, 한 번만)
- `.venv/` 는 `.gitignore` 에 있어 저장소에 올라가지 않는다

## 노트북 `01_CLIP_실습.ipynb`

실행 : **Run → Run All Cells**. 첫 실행은 모델 파일을 받는다(네트워크에 따라 다름). 받은 뒤 전체 약 30초.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 — 모델 · 37품종 후보 · 문구 템플릿 · 내 문장/사진 | – |
| 2 | 모델 불러오기 · 사진 목록 | – |
| 3 | 이미지 타워 ②~⑦ 텐서 모양을 층마다 찍고, 모델 한 줄 결과와 같은지 확인 | 이미지 타워 · 텐서의 흐름 |
| 4 | 텍스트 타워 — 토큰 번호 · 77칸 · [EOS] 가 대표 벡터인지 | 텍스트 타워 · 텐서의 흐름 |
| 5 | 사진 150장 → `[150 × 512]` | – |
| 6 | 사진 4 · 캡션 4 의 N×N 채점표와 양방향 손실(배율 ×14.3 · ×100) | 정렬·손실 — N×N 채점표 계산 |
| 7 | 제로샷 분류 — 후보 = 37품종 이름 문장 | CLIP의 활용과 한계 2번 |
| 8 | 문구 비교와 앙상블 | CLIP의 활용과 한계 2번 |
| 9 | 검색 — 후보 = 사진 150장 | CLIP의 활용과 한계 1번 |
| 10 | 한계 ① 관계 · 결합(직접 그린 도형) | CLIP의 활용과 한계 3번 |
| 11 | 한계 ② 부정 — "안전모 미착용" 검색 | CLIP의 활용과 한계 3번 |
| 12 | 한계 ③ 개수 | CLIP의 활용과 한계 3번 |
| 13 | 내 사진 · 내 문장으로 | – |

## 실측 (강사 PC)

`openai/clip-vit-base-patch32` · transformers 5.17.0 · Windows 강사 PC.

| 경로 | 날짜 | 장치 | 셀 5 사진 150장 | 결과 숫자 |
|---|---|---|---|---|
| ⓪ 4-1 Jupyter 그대로 | 2026-09-30 | CPU · 셀 2 가 설치한 torch 2.14.0+cpu | 4.3s | ①과 셀 1~13 출력이 같음(장치 · 시간 · 설치 안내 줄 제외). 설치 포함 전체 1분 41초(`ai4-lab/jupyter` 새 컨테이너) |
| ① 도커 CPU | 2026-09-30 | CPU(스레드 24) · torch 2.14.0+cpu | 6.0s | 아래 표 |
| ③ 직접 설치 · NVIDIA | 2026-09-30 | RTX 4080 · torch 2.14.0+cu130 | 1.9s | ①과 셀 2~13 출력이 글자까지 같음(장치 · 시간 줄 제외) |
| ② 도커 GPU | 2026-09-30 | – | – | 이미지 빌드 · CUDA 13.0 판 확인까지. **실행은 미검증** — 이 PC 의 Docker Desktop 이 `--gpus all` 에서 NVIDIA 공식 이미지로도 실패(`nvidia-container-cli` 훅 오류) |
| ④ 직접 설치 · 맥 | – | – | – | **미검증**(맥 없음) |

아래 결과는 ① 기준(①과 ③ 동일).

| 셀 | 결과 |
|---|---|
| 3 | 입력 `(1,3,224,224)` 150,528개 → Conv2d `(768,3,32,32)` → `(1,768,7,7)` = 조각 49개 → `(1,50,768)` → 블록 12개 통과 후에도 `(1,50,768)` → [CLS] `(1,768)` → 투영 `(1,512)` → 길이 1. 모델 한 줄 결과와 차이 0 |
| 4 | "a red car" = 49406 · 320 · 736 · 1615 · 49407. 대표 벡터 = [EOS] 행(True). "빨간 자동차" 한글 5글자 → 10조각 |
| 6 | 코사인 대각선 0.291~0.331 · 그 밖 0.154~0.209. 손실 ×14.3 0.390 · ×100 0.000 (학습된 배율 100.0) |
| 7 | 37후보에서 35/40 = 0.875. 틀림 : 샴→버만 2 · 페르시안→버만 · 메인쿤 · 벵갈→아비시니안. 개/고양이 두 칸 분류기 1.000 |
| 8 | 맞힌 비율 : 라벨 단어만 0.775 · "a photo of a {}." 0.875 · "…, a type of pet." 0.925 · 문장 8개 평균 0.900 |
| 10 | 관계 0.40 · 결합 0.10 (찍기 0.50), 두 문장 코사인 차 평균 0.0008 · 0.0054. 대조군(도형 vs 개) 1.00. 결합은 빨강 원 그림 10장 모두에 "파랑 원 + 빨강 네모"를 고름 |
| 11 | 상위 10 중 맞는 비율 : "wearing a hard hat" 1.0 · "without a hard hat" 0.0 · "not wearing a hard hat" 0.0 · "with bare heads" 0.3 (무작위 0.5) |
| 12 | 원 2~6개 개수 0.37 (찍기 0.20). 예측이 3 · 5 로 몰림 |

## 사진 `images/` (150장, 1.9MB)

| 그룹 | 수 | 원본 | 라이선스 |
|---|---|---|---|
| pet | 8품종 × 5 | [timm/oxford-iiit-pet](https://huggingface.co/datasets/timm/oxford-iiit-pet) test 분할(시바 · 사모예드 · 퍼그 · 비글 · 치와와 · 샴 · 페르시안 · 벵갈) | CC BY-SA 4.0 |
| hardhat | 미착용 20 · 착용 20 | [keremberke/hard-hat-detection](https://huggingface.co/datasets/keremberke/hard-hat-detection) test · valid (Roboflow Hard Hats) | CC BY 4.0 |
| shape_relation · shape_binding · shape_count | 20 · 20 · 30 | `tools/prepare_images.py` 가 그린 도형 | 이 저장소 |

- 안전모 사진은 **두 쪽 모두 현장 사진 출처**에서만 골랐다. 이 데이터셋의 미착용 사진 대부분은 실내 웹캠 셀카라, 섞으면 CLIP 이 안전모가 아니라 배경으로 가른다
- 미착용 = 사진 속 전원 미착용, 착용 = 전원 착용. 널리 알려진 인물이 식별되는 사진과 중복 사진은 뺐다(`tools/prepare_images.py` 의 `EXCLUDE`)
- 파일마다 원본 파일 이름이 `images.csv` 의 `source_id` 에 있다

## 문제 해결

| 증상 | 해결 |
|---|---|
| 셀 2 가 `PyTorch 가 없어 … 설치 중` 에서 1~3분 멈춤 | 정상. 처음 한 번 PyTorch 를 받는 중. 다음부터는 바로 넘어간다 |
| 셀 2 설치가 `pip` 오류로 실패 | 인터넷 연결 확인. 사내망이면 ① 도커(빌드 때 받음) 또는 ③ · ④ 로 미리 설치 |
| 셀 2 에서 `DEVICE = '…' 를 이 환경에서 쓸 수 없다` | 셀 1 의 `DEVICE` 를 메시지의 "쓸 수 있는 장치" 중 하나로(또는 `auto`) |
| ② 에서 `could not select device driver` · `nvidia-container-cli` 오류 | 도커가 GPU 를 못 넘긴다 → ① 로 되돌리거나(`docker compose up -d --build`) ③ 직접 설치 |
| `ai4-lab/jupyter` 이미지가 없다는 빌드 오류 | 4-1 에서 `docker compose up -d --build` 를 먼저 |
| 셀 2 에서 모델을 못 받음 | 인터넷 연결 확인. 받은 뒤에는 볼륨에 남아 다시 받지 않는다 |
| 8889 포트를 이미 쓰는 중 | `CLIP_PORT=8890 docker compose up -d` 후 http://localhost:8890 |
| 내 사진을 쓰고 싶다 | `5-1/images` 에 넣고 셀 1 의 `MY_IMAGE` 에 파일 이름 |

## 강사용 — 사진 · 노트북 다시 만들기

```bash
# 사진 : hard-hat-detection 의 data/test.zip · data/valid.zip 을 받아 둔 뒤
python tools/prepare_images.py --hardhat-zip <test.zip> --hardhat-zip <valid.zip>
# 노트북 : 셀 내용을 tools/build_notebook.py 에서 고치고
python tools/build_notebook.py
```
