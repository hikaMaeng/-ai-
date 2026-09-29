# 5강 실습 — CLIP : 사진과 글을 한 공간에서 재기

> 현대 AI의 원리와 구조 #5-1 이미지 생성 실습 · 뉴런데브클래스
> 전제 : [4-1](../4-1/) 의 도커 환경을 한 번 띄운 적이 있다(`ai4-lab/jupyter` 이미지 · `hfcache` 볼륨). **LM Studio 는 필요 없다.**
> 실습 화면 : Jupyter **http://localhost:8889** → `5-1/notebooks/01_CLIP_실습.ipynb`

장표 「CLIP 대조학습」 · 「CLIP의 활용과 한계」 · 「이미지·텍스트 임베딩 파이프라인」(이미지 타워 · 텍스트 타워 · 정렬·손실)을
CLIP ViT-B/32 로 직접 돌려, 장표의 텐서 모양과 주장(제로샷 분류 · 검색 · 관계/부정/개수의 한계)을 숫자로 확인한다.

```
5-1/
├─ docker-compose.yml     4-1 스택 옆에 CLIP 용 Jupyter(ai4-clip, 포트 8889) 하나를 더 띄운다
├─ clip/Dockerfile        ai4-lab/jupyter + CPU PyTorch + transformers
├─ notebooks/
│  └─ 01_CLIP_실습.ipynb   셀 13개. 코드 셀 첫 줄 = "# 셀 N · 제목", 설정은 셀 1
├─ images/                실습 사진 150장 + images.csv(파일 · 그룹 · 라벨 · 출처 · 라이선스)
└─ tools/                 강사용 — 수강생은 실행하지 않는다
   ├─ prepare_images.py   images/ 를 다시 만드는 스크립트(원본 데이터셋에서 선별)
   └─ build_notebook.py   노트북 생성기(셀 내용은 여기서 고친다)
```

## 기동

```bash
cd ./-ai-/5-1
docker compose up -d --build
```

- 4-1 의 `ai4-lab/jupyter` 이미지 위에 CPU 전용 PyTorch(`torch==2.14.*`, CPU 휠) · `transformers==5.17.*` 만 더한다. 4-1 · 4-2 이미지는 그대로다
- 결과 이미지 `ai4-lab/jupyter-clip` 약 1.7GB. 첫 빌드는 PyTorch 를 받느라 수 분 걸린다
- 컨테이너 `ai4-clip` 은 4-1 과 같은 compose 프로젝트(`ai4-lab`)에 들어가고 `hfcache` 볼륨을 같이 쓴다 → CLIP 모델(약 600MB)을 한 번 받으면 남는다
- 포트 8889 는 내 PC(127.0.0.1)에서만 열린다. 겹치면 `CLIP_PORT=8890 docker compose up -d`
- GPU 는 쓰지 않는다

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

2026-09-29 · Windows · Docker · CPU(스레드 24) · torch 2.14.0+cpu · transformers 5.17.0 · `openai/clip-vit-base-patch32` · 모델 캐시가 있는 상태로 전체 실행 28초.

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
