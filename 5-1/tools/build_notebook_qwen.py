"""5-1 실습 노트북 02 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook_qwen.py   → ../notebooks/02_Qwen3VL_임베딩_실습.ipynb (출력 없이)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / '02_Qwen3VL_임베딩_실습.ipynb'
cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': s.strip('\n').splitlines(keepends=True)})


md('''
# 실습 · 멀티모달 임베딩 — Qwen3-VL-Embedding-2B

장표의 VL 임베딩 섹션(「CLIP 다음의 검색 벡터」 · 「구조 및 학습」 · 「활용 사례」 · 「사용법」)을 직접 돌려 본다. 01 CLIP 실습과 **같은 사진 150장**을 쓰고, 문서 스크린샷 6장을 더한다.
CLIP 은 사진과 글을 **따로** 읽는 두 타워, 이 모델은 멀티모달 LLM(Qwen3-VL) **하나가 함께** 읽어 마지막 토큰의 벡터를 쓴다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | 모델 불러오기 · 임베딩 함수 | – |
| 3 | 입력 구조 — 지시문 · 이미지 토큰 · 마지막 토큰 | VL 임베딩 — 구조 및 학습 1번 |
| 4 | 사진 150장 임베딩 `[150 × 2048]` | – |
| 5 | 사진 3 · 캡션 3 코사인 표 — CLIP 과 같은 사진 | CLIP 대조학습 3번 |
| 6 | 제로샷 분류 37품종 — 지시문을 넣으면 | 구조 및 학습 1번 "지시문이 벡터를 바꿈" |
| 7 | 한국어로 검색 | CLIP 다음의 검색 벡터 (01 셀 4 의 한글 쪼개짐) |
| 8 | CLIP 한계 다시 재기 — 관계 · 결합 · 부정 · 개수 | CLIP의 활용과 한계 3번 · 활용 사례 5번 |
| 9 | 섞인 입력 — 사진 + 글을 벡터 하나로 | 활용 사례 3번 |
| 10 | 문서 스크린샷 검색 | 활용 사례 1번 · CLIP 다음의 검색 벡터 1번 |
| 11 | 차원 자르기(MRL) | 구조 및 학습 3번 |
| 12 | 대가 — 크기 · 속도를 CLIP 과 비교 | 사용법 3번 |
| 13 | 사진으로 찾기 — 사진 → 사진 | 활용 사례 2번 |
| 14 | 정답이 없을 때 — 그래도 가장 가까운 것을 돌려줌 | 활용 사례 5번 |
| 15 | 내 사진 · 내 문장으로 | – |

- 실행 : **Run → Run All Cells**. 첫 실행은 모델 약 4GB 를 받는다. 받은 뒤 CPU 에서 전체 약 5분(사진 · 문서 임베딩이 대부분)
- 메모리 : 모델을 float32 로 올리면 약 9GB. 모자라면 셀 1 `DTYPE = "bfloat16"`(약 4.5GB, 숫자가 소수 셋째 자리에서 조금 달라짐)
- 장치 · 설치는 01 과 같다 : 셀 1 `DEVICE`, PyTorch 가 없는 Jupyter(4-1 의 8888)면 셀 2 가 처음 한 번 설치
- 사진 출처 · 라이선스 : `5-1/images/images.csv`. 문서 스크린샷 `5-1/docs` 는 가상의 서식(`tools/make_docs.py`)
''')

md('''
### 셀 1 · 설정
- `MODEL` : Qwen3-VL-Embedding **2B**(작은 쪽. 8B 도 있음) — 출력 2048차원, 32개 이상 언어, 지시문 입력
- `DTYPE` : `float32`(기본, 약 9GB) · `bfloat16`(약 4.5GB)
- `MAX_PIXELS` : 사진 한 장을 몇 픽셀까지 쓸지. 32×32 픽셀 = 이미지 토큰 1개. 기본은 모델 권장값(토큰 1800개까지)
- `INS_*` : 지시문(system 자리에 들어가는 한 문장). 모델 기본값은 `"Represent the user's input."`
- `MY_*` : 셀 15 에서 바꿔 본다
''')
code('''
# 셀 1 · 설정
DEVICE     = "auto"      # "auto" · "cuda"(NVIDIA GPU) · "mps"(맥 Metal) · "cpu"
DTYPE      = "float32"   # "float32" · "bfloat16"(메모리 절반)
MODEL      = "Qwen/Qwen3-VL-Embedding-2B"
IMAGE_DIR  = "auto"      # auto = 이 노트북 옆의 ../images
DOC_DIR    = "auto"      # auto = 이 노트북 옆의 ../docs
MAX_PIXELS = 1800 * 32 * 32

INS_DEFAULT  = "Represent the user's input."
INS_CLASSIFY = "Classify the pet breed in the image."
INS_DOC      = "Retrieve document images that answer the user's query."
INS_PASSAGE  = "Retrieve the text passage that answers the user's query about the image."

# 셀 15 : 내 사진 · 내 문장 (images 폴더 파일 이름 또는 /work 아래 경로)
MY_IMAGE = "hh_on_01.jpg"
MY_TEXTS = ["안전모를 쓴 건설 작업자", "회의실에서 정장을 입은 사람들", "잔디 위의 개"]
MY_INSTRUCTION = INS_DEFAULT
''')

md('''
### 셀 2 · 모델 불러오기 · 임베딩 함수
- PyTorch · torchvision · transformers 가 없으면 이 Jupyter 에 설치한다(처음 한 번). torchvision 은 Qwen3-VL 프로세서가 씀
- 모델 몸체는 멀티모달 LLM `Qwen3VLModel`(이미지 인코더 + 언어 모델). 다음 토큰을 고르는 머리(lm_head)는 쓰지 않는다
- `embed()` : 입력마다 대화 한 개를 만든다 → [system : 지시문] [user : 사진 · 글] [assistant 차례]
  → 모델 통과 → **마지막 토큰의 출력 벡터** → 길이 1로 정규화
  - 공식 구현(QwenLM/Qwen3-VL-Embedding)과 같은 계산. 사진 크기 맞춤(32픽셀 배수)도 공식과 같다
''')
code('''
# 셀 2 · 모델 불러오기 · 임베딩 함수
import math, os, time, warnings
from pathlib import Path
for k, v in {"HF_HUB_VERBOSITY": "error", "TRANSFORMERS_VERBOSITY": "error",
             "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}.items():
    os.environ.setdefault(k, v)
if os.access("/cache", os.W_OK):              # 도커 : 모델을 hfcache 볼륨에
    os.environ.setdefault("HF_HOME", "/cache/hf")
warnings.filterwarnings("ignore", message="IProgress not found")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import importlib, importlib.util, shutil, subprocess, sys
def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--root-user-action=ignore", *args])
if importlib.util.find_spec("torch") is None:
    gpu = DEVICE == "cuda" or (DEVICE == "auto" and shutil.which("nvidia-smi") is not None)
    if sys.platform == "darwin":
        kind, index = "맥 기본판(Metal)", []
    elif gpu:
        kind, index = "CUDA 판(수 GB)", [] if sys.platform.startswith("linux") else ["--index-url", "https://download.pytorch.org/whl/cu130"]
    else:
        kind, index = "CPU 판(약 200MB)", ["--index-url", "https://download.pytorch.org/whl/cpu"]
    print(f"PyTorch 가 없어 이 Jupyter 에 설치 중 · {kind} · 처음 한 번만 (1~3분) ...")
    pip_install(*index, "torch==2.14.*")
    importlib.invalidate_caches()
# Qwen3-VL 프로세서는 torchvision 이 있어야 만들어진다(01 CLIP 은 필요 없음) → 설치된 torch 와 같은 판(cpu · cu130 · 기본)으로
if importlib.util.find_spec("torchvision") is None:
    import importlib.metadata
    local = importlib.metadata.version("torch").partition("+")[2]
    print(f"torchvision 설치 중 · 처음 한 번만 · torch 와 같은 판({local or '기본'}) ...")
    pip_install(*(["--index-url", f"https://download.pytorch.org/whl/{local}"] if local else []), "torchvision==0.29.*")
if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("safetensors") is None:
    print("transformers 설치 중 · 처음 한 번만 ...")
    pip_install("transformers==5.17.*", "safetensors")
importlib.invalidate_caches()

import torch
import torch.nn.functional as F
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()

have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "AppleGothic", "Malgun Gothic") if f in have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 110})
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
avail = {"cuda": torch.cuda.is_available(),
         "mps": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
         "cpu": True}
want = next(d for d in ("cuda", "mps", "cpu") if avail[d]) if DEVICE == "auto" else DEVICE
if not avail.get(want, False):
    raise RuntimeError(f"DEVICE = '{DEVICE}' 를 이 환경에서 쓸 수 없다. 쓸 수 있는 장치 : {[d for d, ok in avail.items() if ok]}")
DEV = torch.device(want)
name = torch.cuda.get_device_name(DEV) if DEV.type == "cuda" else ("Apple GPU (Metal)" if DEV.type == "mps" else "CPU")
print(f"장치 {DEV} · {name} · torch {torch.__version__} · {DTYPE}")

def find_dir(value, sub, marker):
    if value != "auto":
        return value
    d = next((str(p) for p in (Path.cwd().parent / sub, Path(f"/work/5-1/{sub}"), Path.cwd() / "5-1" / sub) if (p / marker).exists()), None)
    assert d, f"5-1/{sub} 폴더를 찾지 못했다 → 셀 1 에 경로를 적는다"
    return d
IMAGE_DIR = find_dir(IMAGE_DIR, "images", "images.csv")
DOC_DIR = find_dir(DOC_DIR, "docs", "docs.csv")

t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL, padding_side="right")
model = Qwen3VLModel.from_pretrained(MODEL, dtype=getattr(torch, DTYPE)).eval().to(DEV)
print(f"모델 불러오기 {time.time() - t0:.1f}s · 파라미터 {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

meta = pd.read_csv(Path(IMAGE_DIR) / "images.csv")
docs = pd.read_csv(Path(DOC_DIR) / "docs.csv")
print(meta.groupby("group").size().to_string(), f"\\n문서 스크린샷 {len(docs)}장")

FACTOR, MIN_PIXELS = 32, 4 * 32 * 32
def fit_size(h, w):
    """사진을 32픽셀 배수로 맞춘 크기 (공식 구현의 smart_resize 와 같은 규칙)"""
    hb, wb = max(FACTOR, round(h / FACTOR) * FACTOR), max(FACTOR, round(w / FACTOR) * FACTOR)
    if hb * wb > MAX_PIXELS:
        b = math.sqrt(h * w / MAX_PIXELS); hb, wb = math.floor(h / b / FACTOR) * FACTOR, math.floor(w / b / FACTOR) * FACTOR
    elif hb * wb < MIN_PIXELS:
        b = math.sqrt(MIN_PIXELS / (h * w)); hb, wb = math.ceil(h * b / FACTOR) * FACTOR, math.ceil(w * b / FACTOR) * FACTOR
    return hb, wb

def path_of(name):
    if "/" in str(name) or "\\\\" in str(name):
        return Path(name)
    return Path(DOC_DIR if str(name).startswith("doc_") else IMAGE_DIR) / name

def load(name):
    im = Image.open(path_of(name)).convert("RGB")
    h, w = fit_size(im.height, im.width)
    return im.resize((w, h))

def prepare(items, ins=INS_DEFAULT):
    """items : [{"image": 파일, "text": 문장}, ...] — 둘 중 하나 또는 둘 다"""
    convs, imgs = [], []
    for it in items:
        content = []
        if it.get("image"):
            content.append({"type": "image"}); imgs.append(load(it["image"]))
        if it.get("text"):
            content.append({"type": "text", "text": it["text"]})
        convs.append([{"role": "system", "content": [{"type": "text", "text": it.get("ins", ins)}]},
                      {"role": "user", "content": content}])
    text = proc.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)
    return proc(text=text, images=imgs or None, padding=True, do_resize=False, return_tensors="pt")

def embed(items, ins=INS_DEFAULT, batch=8):
    out = []
    for i in range(0, len(items), batch):
        inp = prepare(items[i:i + batch], ins).to(DEV)
        h = model(**inp).last_hidden_state                       # [B, 길이, 2048]
        m = inp["attention_mask"]
        last = m.shape[1] - 1 - m.flip(1).argmax(1)              # 칸마다 마지막 실제 토큰 위치
        out.append(F.normalize(h[torch.arange(h.shape[0]), last].float(), dim=-1).cpu())
    return torch.cat(out)

texts  = lambda qs, ins=INS_DEFAULT: embed([{"text": q} for q in qs], ins)
images = lambda fs, ins=INS_DEFAULT: embed([{"image": f} for f in fs], ins)

def show(names, titles=None, cols=8, size=1.6):
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * (size + 0.35)))
    for ax in np.array(axes).reshape(-1):
        ax.axis("off")
    for i, n in enumerate(names):
        ax = np.array(axes).reshape(-1)[i]
        ax.imshow(Image.open(path_of(n)).convert("RGB")); ax.set_title(titles[i] if titles else n, fontsize=7)
    plt.tight_layout(); plt.show()
''')

md('''
### 셀 3 · 입력 구조 — 지시문 · 이미지 토큰 · 마지막 토큰
장표 「VL 임베딩 — 구조 및 학습」 1번 — "입력 : 지시문 + 글·사진을 섞어서 → VLM 몸체 통과", "끝 토큰 자리의 출력 1개만 꺼냄 → L2 정규화".
- 모델에 실제로 들어가는 글(대화 틀)을 그대로 찍는다. `<|image_pad|>` 한 칸 = 이미지 토큰 하나
- 사진 크기 → 16픽셀 패치 격자 → 2×2 패치를 하나로 합침 → 이미지 토큰 수
- CLIP 은 사진 벡터 · 글 벡터를 따로 만들지만, 여기서는 사진과 글이 **한 입력열**
''')
code('''
# 셀 3 · 입력 구조
item = {"image": "pet_pug_1.jpg", "text": "이 개의 품종은?"}
inp = prepare([item])
raw = Image.open(path_of(item["image"]))
t, gh, gw = inp["image_grid_thw"][0].tolist()
print(f"사진 {raw.width}×{raw.height} → 32의 배수로 {gw * 16}×{gh * 16} → 16픽셀 패치 {gh}×{gw} = {gh * gw}개 → 2×2 합치기 → 이미지 토큰 {gh * gw // 4}개")
print(f"입력열 전체 {inp['input_ids'].shape[1]} 토큰 = 지시문 · 대화 틀 · 이미지 토큰 · 글 토큰\\n")

ids = inp["input_ids"][0].tolist()
pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
shown = proc.tokenizer.decode(ids).replace("<|image_pad|>" * ids.count(pad), f"<|image_pad|> × {ids.count(pad)}")
print("모델에 들어가는 글 :\\n" + shown)
print(f"마지막 토큰 = {proc.tokenizer.decode(ids[-1:])!r} → 이 자리의 출력 벡터가 임베딩")

h = model(**inp.to(DEV)).last_hidden_state
print(f"\\n출력 {tuple(h.shape)} = [입력 1개, 토큰 {h.shape[1]}개, 폭 {h.shape[2]}]  → 마지막 행 [{h.shape[2]}] → 길이 1로 정규화")
v = embed([item])[0]
print(f"임베딩 {tuple(v.shape)} · 길이 {v.norm():.4f}")
''')

md('''
### 셀 4 · 사진 150장 임베딩
- 01 CLIP 실습과 같은 150장 → `[150 × 2048]`. CPU 에서 약 2분 (CLIP 은 6초)
- 사진마다 크기가 달라 이미지 토큰 수도 다르다 → 사진 크기를 224×224 로 누르지 않는다
''')
code('''
# 셀 4 · 사진 150장 임베딩
t0 = time.time()
V = images(list(meta.file))
emb = dict(zip(meta.file, V))
tok = [int(prepare([{"image": f}])["image_grid_thw"][0].prod()) // 4 for f in meta.file[::10]]
print(f"사진 {len(meta)}장 → {tuple(V.shape)}  {time.time() - t0:.1f}s")
print(f"이미지 토큰 수(15장 표본) 최소 {min(tok)} · 최대 {max(tok)}  — CLIP 은 모든 사진이 49 + [CLS]")
''')

md('''
### 셀 5 · 사진 3 · 캡션 3 코사인 표
장표 「CLIP 대조학습」 3번의 N×N 행렬 — 01 실습 셀 6 과 **같은 사진 · 같은 캡션**.
- 학습 방식이 같은 대조학습(InfoNCE)이므로 제 짝(대각선)이 밝다
- CLIP 실측(01 셀 6) : 대각선 0.29~0.33 · 그 밖 0.15~0.21. 코사인의 절대 크기는 모델마다 달라 **순서**로 비교
''')
code('''
# 셀 5 · 사진 3 · 캡션 3 코사인 표
files = ["pet_shiba_inu_1.jpg", "pet_persian_1.jpg", "hh_on_01.jpg"]
caps  = ["a photo of a shiba inu dog.", "a photo of a persian cat.", "a construction worker wearing a hard hat."]
S = torch.stack([emb[f] for f in files]) @ texts(caps).T
print(pd.DataFrame(S.numpy(), index=["사진 시바", "사진 페르시안", "사진 안전모"], columns=["캡션 시바", "캡션 페르시안", "캡션 안전모"]).round(3))
off = S[~torch.eye(3, dtype=bool)]
print(f"\\n대각선 {S.diag().min():.3f}~{S.diag().max():.3f} · 그 밖 {off.min():.3f}~{off.max():.3f}")
''')

md('''
### 셀 6 · 제로샷 분류 37품종 — 지시문을 넣으면
장표 「VL 임베딩 — 구조 및 학습」 1번 "지시문이 벡터를 바꿈 : 기본 \"Represent the user's input.\"" (3강 instruction 임베딩).
- 후보 = 37품종 이름. 사진 40장 × 후보 37 → 행마다 최고점이 답 (01 셀 7 과 같은 문제)
- ① 지시문 기본값 ② 사진 · 후보 모두 `INS_CLASSIFY` 로 다시 임베딩
- CLIP 실측(01 셀 7 · 8) : "a photo of a {}." 0.875 · 앙상블 0.900 · 최고 0.925
''')
code('''
# 셀 6 · 제로샷 분류 37품종
BREEDS_37 = ["abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle", "bengal",
             "birman", "bombay", "boxer", "british_shorthair", "chihuahua", "egyptian_mau", "english_cocker_spaniel",
             "english_setter", "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin", "keeshond",
             "leonberger", "maine_coon", "miniature_pinscher", "newfoundland", "persian", "pomeranian", "pug",
             "ragdoll", "russian_blue", "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu", "siamese",
             "sphynx", "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"]
NAMES = [n.replace("_", " ") for n in BREEDS_37]
pets = meta[meta.group == "pet"].reset_index(drop=True)

def accuracy(P, W):
    pred = [BREEDS_37[i] for i in (P @ W.T).argmax(dim=1).tolist()]
    return float(np.mean([p == t for p, t in zip(pred, pets.label)])), pred

P0, W0 = torch.stack([emb[f] for f in pets.file]), texts(NAMES)
P1, W1 = images(list(pets.file), INS_CLASSIFY), texts(NAMES, INS_CLASSIFY)
acc0, _ = accuracy(P0, W0)
acc1, pred1 = accuracy(P1, W1)
print(f"① 지시문 기본값          {acc0:.3f}")
print(f"② 지시문 {INS_CLASSIFY!r}  {acc1:.3f}")
print("②에서 틀린 사진 :", ", ".join(f"{t}→{p}" for t, p in zip(pets.label, pred1) if t != p) or "없음")
''')

md('''
### 셀 7 · 한국어로 검색
- 01 셀 9 의 영어 질의를 **한국어로** 그대로 옮겼다. CLIP 은 한글을 잘게 쪼개 읽어(01 셀 4) 한국어 검색이 사실상 안 된다
- 후보 = 사진 150장 → 상위 5장
''')
code('''
# 셀 7 · 한국어로 검색
ALL = torch.stack([emb[f] for f in meta.file])
for q in ["하얀 털이 복슬복슬한 개", "주황색 안전 조끼를 입은 작업자", "점무늬 털의 고양이"]:
    s = (texts([q]) @ ALL.T)[0]; top = s.topk(5)
    names = [meta.file[i] for i in top.indices.tolist()]
    print(f"{q!r}  상위 5 코사인 {top.values.numpy().round(3)}\\n  {names}")
    show(names, [f"{n}\\n{v:.3f}" for n, v in zip(names, top.values.tolist())], cols=5, size=1.9)
''')

md('''
### 셀 8 · CLIP 한계 다시 재기 — 관계 · 결합 · 부정 · 개수
장표 「CLIP의 활용과 한계」 3번을 같은 사진 · 같은 문장으로 다시 잰다. 01 셀 10~12 와 같은 방법.
「VL 임베딩 — 활용 사례」 5번 "관계 · 동작은 벡터 1개에서 흐려질 수 있음" 을 확인하는 셀이기도 하다.
- 모델이 커지고 LLM 이 글을 읽으면 풀리는 것과 **그대로 남는 것**을 가른다
- CLIP 열은 01 노트북 실측값
- 끝에 부정의 우회 : "안전모를 쓴 작업자" 점수가 **가장 낮은** 10장 — 부정은 문장이 아니라 코드(정렬 방향)로
''')
code('''
# 셀 8 · CLIP 한계 다시 재기
def two_way(group, pair, first):
    g = meta[meta.group == group].reset_index(drop=True)
    pick = (torch.stack([emb[f] for f in g.file]) @ texts(pair).T).argmax(dim=1).numpy()
    return float(((pick == 0) == (g.label == first)).mean())

hh = meta[meta.group == "hardhat"].reset_index(drop=True)
H = torch.stack([emb[f] for f in hh.file])
def top10(q, want):
    idx = (texts([q]) @ H.T)[0].topk(10).indices.tolist()
    return float((hh.label[idx] == want).mean())

cnt = meta[meta.group == "shape_count"].reset_index(drop=True)
words = ["two", "three", "four", "five", "six"]
pred = [2 + i for i in (torch.stack([emb[f] for f in cnt.file]) @ texts([f"a picture of {w} black circles" for w in words]).T).argmax(dim=1).tolist()]

rows = [
    ("관계 : 빨강/파랑 네모 위아래", "찍기 0.50", 0.40, two_way("shape_relation", ["a red square above a blue square", "a blue square above a red square"], "red above blue")),
    ("결합 : 원 · 네모의 색", "찍기 0.50", 0.10, two_way("shape_binding", ["a red circle and a blue square", "a blue circle and a red square"], "red circle and blue square")),
    ("부정 : without a hard hat", "상위 10 중 미착용", 0.00, top10("a construction worker without a hard hat", "no-hardhat")),
    ("부정 : 안전모를 쓰지 않은 작업자", "상위 10 중 미착용", None, top10("안전모를 쓰지 않은 작업자", "no-hardhat")),
    ("대조 : 안전모를 쓴 작업자", "상위 10 중 착용", None, top10("안전모를 쓴 작업자", "hardhat")),
    ("개수 : 검은 원 2~6개", "찍기 0.20", 0.37, float(np.mean([p == t for p, t in zip(pred, cnt.label.astype(int))]))),
]
print(pd.DataFrame(rows, columns=["시험", "기준", "CLIP (01 실측)", "Qwen3-VL-Emb"]).to_string(index=False, na_rep="(한글 불가)", float_format=lambda x: f"{x:.2f}"))
s_on = (texts(["안전모를 쓴 작업자"]) @ H.T)[0]
low = s_on.topk(10, largest=False).indices.tolist()
print(f"\\n우회 : '안전모를 쓴 작업자' 점수가 가장 낮은 10장 중 미착용 {float((hh.label[low] == 'no-hardhat').mean()):.2f}")
print("\\n개수 — 실제 × 예측"); print(pd.crosstab(pd.Series(cnt.label.astype(int).tolist(), name="실제"), pd.Series(pred, name="예측")))
''')

md('''
### 셀 9 · 섞인 입력 — 사진 + 글을 벡터 하나로
장표 「VL 임베딩 — 활용 사례」 3번 "사진 + 글로 찾기 — 섞인 질의". CLIP 은 사진 벡터 · 글 벡터가 따로라 섞인 질의를 벡터 하나로 못 만든다(「CLIP 다음의 검색 벡터」 1번).
- 글 문서 8개 = 주제 4(안전모 · 퍼그 · 사모예드 · 샴) × 종류 2(관리 수칙 · 소개)
- 질의 = [사진] + "이것을 다룰 때 지켜야 할 수칙" 처럼 **무엇에 대해**는 사진이, **어떤 글을**은 글이 말한다
- 네 가지 질의를 비교 : 사진만 · 글만 · 두 벡터 평균(CLIP 식 우회) · 사진과 글을 한 입력으로
''')
code('''
# 셀 9 · 섞인 입력
PASSAGES = {
    ("안전모", "수칙"): "건설 현장 안전모 착용 수칙: 현장 출입 전 안전모를 쓰고 턱끈을 조인다. 균열이나 변형이 있는 안전모는 즉시 교체한다.",
    ("안전모", "소개"): "안전모 소개: 머리를 낙하물과 충돌로부터 보호하는 보호구. 외피는 ABS나 FRP, 안쪽에 충격 흡수 라이너가 있다.",
    ("퍼그", "수칙"): "퍼그 관리 수칙: 코가 짧은 단두종이라 더운 날 산책을 줄이고 호흡을 살핀다. 얼굴 주름 사이는 매일 닦아 준다.",
    ("퍼그", "소개"): "퍼그 소개: 중국 원산의 소형견. 주름진 얼굴, 말린 꼬리, 짧은 털이 특징이다.",
    ("사모예드", "수칙"): "사모예드 관리 수칙: 이중모라 털갈이 때는 매일 빗질한다. 더위에 약해 여름에는 시원한 곳에서 지내게 한다.",
    ("사모예드", "소개"): "사모예드 소개: 시베리아 원산의 썰매견. 풍성한 흰 털과 웃는 듯한 입꼬리가 특징이다.",
    ("샴", "수칙"): "샴 고양이 관리 수칙: 사람을 잘 따라 혼자 두는 시간을 줄인다. 털이 짧아 주 1회 빗질이면 충분하다.",
    ("샴", "소개"): "샴 고양이 소개: 태국 원산. 밝은 몸에 얼굴·귀·발·꼬리 끝이 짙은 포인트 무늬, 파란 눈이 특징이다.",
}
PHOTO = {"안전모": "hh_off_03.jpg", "퍼그": "pet_pug_2.jpg", "사모예드": "pet_samoyed_2.jpg", "샴": "pet_siamese_2.jpg"}
ASK = {"수칙": "이것을 다룰 때 지켜야 할 수칙", "소개": "이것이 무엇인지 소개하는 글"}
keys = list(PASSAGES)
K = texts(list(PASSAGES.values()))

rows = []
for subj, photo in PHOTO.items():
    for kind, ask in ASK.items():
        qi = embed([{"image": photo}], INS_PASSAGE)[0]
        qt = embed([{"text": ask}], INS_PASSAGE)[0]
        qm = embed([{"image": photo, "text": ask}], INS_PASSAGE)[0]
        qa = F.normalize(qi + qt, dim=-1)
        got = [" · ".join(keys[int((q @ K.T).argmax())]) for q in (qi, qt, qa, qm)]
        rows.append([f"[{subj} 사진] + {ask}", f"{subj} · {kind}"] + got)
r = pd.DataFrame(rows, columns=["질의", "정답", "사진만", "글만", "두 벡터 평균", "한 입력으로"])
print(r.to_string(index=False))
print("\\n맞힌 비율 :", {c: float((r[c] == r["정답"]).mean()) for c in ["사진만", "글만", "두 벡터 평균", "한 입력으로"]})
show(list(PHOTO.values()), list(PHOTO), cols=4, size=1.8)
''')

md('''
### 셀 10 · 문서 스크린샷 검색
장표 「VL 임베딩 — 활용 사례」 1번 "문서 페이지를 사진째 검색 — 글자 추출(OCR) 없이". CLIP 은 224 정사각형으로 눌러 작은 글씨를 못 읽는다(「CLIP 다음의 검색 벡터」 1번).
- 문서 스크린샷 6장(900×1100, 가상의 공정 · 안전 서식)을 **이미지로** 임베딩. 글자를 뽑아내는(OCR) 단계가 없다
- 후보 = 문서 6장 + 사진 150장. 질의는 문서 제목을 그대로 쓰지 않고 바꿔 말한 한국어
- 지시문 기본값과 `INS_DOC` 를 비교. CPU 에서 문서 한 장 약 13초 (이미지 토큰 약 950개)
''')
code('''
# 셀 10 · 문서 스크린샷 검색
t0 = time.time()
D = images(list(docs.file))
print(f"문서 {len(docs)}장 {tuple(D.shape)}  {time.time() - t0:.0f}s · 한 장 이미지 토큰 {int(prepare([{'image': docs.file[0]}])['image_grid_thw'][0].prod()) // 4}개")
show(list(docs.file), list(docs.title), cols=6, size=2.2)

CAND = torch.cat([D, ALL]); cand_names = list(docs.file) + list(meta.file)
QUERIES = ["배관 수압 시험 결과", "용접하기 전에 받아야 하는 허가", "펌프 진동 점검 결과",
           "톨루엔 다룰 때 필요한 보호구", "이번 달 안전 교육 일정", "반응기 온도가 경보치를 넘은 기록"]
for label, ins in [("지시문 기본값", INS_DEFAULT), ("INS_DOC", INS_DOC)]:
    top = (texts(QUERIES, ins) @ CAND.T).argmax(dim=1).tolist()
    hit = [cand_names[t] == docs.file[i] for i, t in enumerate(top)]
    print(f"\\n[{label}] 1위가 정답 문서인 비율 {np.mean(hit):.2f}")
    for q, t, ok in zip(QUERIES, top, hit):
        print(f"  {'O' if ok else 'X'} {q} → {cand_names[t]}")
''')

md('''
### 셀 11 · 차원 자르기(MRL)
3강 MRL — 벡터의 앞쪽 d개만 남기고 다시 길이 1로 맞춘다. 저장 공간 · 검색 속도가 d 에 비례해 준다.
- 셀 6 의 37품종 분류를 d = 2048 → 64 로 잘라 가며 잰다
- 지시문 기본값(①)과 `INS_CLASSIFY`(②) 둘 다
''')
code('''
# 셀 11 · 차원 자르기(MRL)
rows = []
for d in [2048, 1024, 512, 256, 128, 64]:
    cut = lambda X: F.normalize(X[:, :d], dim=-1)
    rows.append((d, f"{d * 4:,} B", accuracy(cut(P0), cut(W0))[0], accuracy(cut(P1), cut(W1))[0]))
print(pd.DataFrame(rows, columns=["차원", "벡터 1개(float32)", "① 기본 지시문", "② INS_CLASSIFY"]).to_string(index=False))
''')

md('''
### 셀 12 · 대가 — 크기 · 속도를 CLIP 과 비교
장표 「VL 임베딩 — 사용법」 3번 "사진만 · 대량 · 실시간 → CLIP 계열" — 크기와 속도가 그 이유.
- 같은 사진 10장을 한 장씩 임베딩하는 시간. CLIP 모델(약 600MB)이 없으면 이 셀에서 받는다
''')
code('''
# 셀 12 · 대가 — 크기 · 속도
from transformers import CLIPModel, CLIPProcessor
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(DEV)
cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sample = list(meta.file[:10])

def per_image(fn):
    fn(sample[0])                                   # 첫 호출(준비 시간)은 빼고 잰다
    t0 = time.time()
    for f in sample:
        fn(f)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / len(sample)

t_clip = per_image(lambda f: clip.get_image_features(**cproc(images=Image.open(path_of(f)).convert("RGB"), return_tensors="pt").to(DEV)))
t_qwen = per_image(lambda f: images([f]))
n_clip = sum(p.numel() for p in clip.parameters()); n_qwen = sum(p.numel() for p in model.parameters())
print(pd.DataFrame([("CLIP ViT-B/32", f"{n_clip / 1e6:.0f}M", 512, f"{t_clip:.3f}s"),
                    ("Qwen3-VL-Embedding-2B", f"{n_qwen / 1e9:.2f}B", 2048, f"{t_qwen:.3f}s")],
                   columns=["모델", "파라미터", "차원", "사진 1장"]).to_string(index=False))
print(f"\\n파라미터 {n_qwen / n_clip:.0f}배 · 사진 1장 시간 {t_qwen / t_clip:.0f}배")
''')

md('''
### 셀 13 · 사진으로 찾기 — 사진 → 사진
장표 「VL 임베딩 — 활용 사례」 2번 "현장 부품 사진 → 카탈로그 · 정비 기록 / 불량 사진 → 비슷한 과거 사례".
- 질의도 사진이다. 사진 벡터와 가장 가까운 사진 5장(자기 자신은 뺌)
- 셀 12 에서 불러온 CLIP 으로도 같은 일을 해 나란히 본다 — 사진 → 사진은 CLIP 도 할 수 있는 일
''')
code('''
# 셀 13 · 사진으로 찾기
def cimg(fs):                                   # transformers 5 : 출력 객체면 pooler_output 이 투영된 사진 벡터
    outs = [clip.get_image_features(**cproc(images=Image.open(path_of(f)).convert("RGB"), return_tensors="pt").to(DEV)) for f in fs]
    return F.normalize(torch.cat([getattr(o, "pooler_output", o).float().cpu() for o in outs]), dim=-1)
ALL_CLIP = cimg(list(meta.file))
label = dict(zip(meta.file, meta.label))
for q in ["hh_off_03.jpg", "pet_pug_2.jpg"]:
    for name, M in [("Qwen3-VL-Emb", ALL), ("CLIP", ALL_CLIP)]:
        s = M @ M[list(meta.file).index(q)]
        s[list(meta.file).index(q)] = -1
        top = s.topk(5)
        names = [meta.file[i] for i in top.indices.tolist()]
        same = sum(label[n] == label[q] for n in names)
        print(f"[{name}] 질의 {q}({label[q]}) → 같은 라벨 {same}/5 : {names}")
    show([q] + names, [f"질의 {q}"] + [f"{n}\\n{label[n]}" for n in names], cols=6, size=1.9)
''')

md('''
### 셀 14 · 정답이 없을 때 — 그래도 가장 가까운 것을 돌려줌
장표 「VL 임베딩 — 활용 사례」 5번 "정답이 없어도 가장 가까운 것을 돌려줌 → 점수 기준선 · 리랭커 · 사람 확인".
- 사진 150장에 **있는 것**과 **없는 것**을 각각 검색한다(지게차 · 용접 불꽃 · 자동차 엔진은 150장에 없다)
- 없는 것도 1위가 나온다. 1위 코사인으로 "있다 / 없다" 를 가를 수 있는지 본다
''')
code('''
# 셀 14 · 정답이 없을 때
PRESENT = ["안전모를 쓴 작업자", "하얀 털이 복슬복슬한 개", "점무늬 털의 고양이", "빨간 네모와 파란 네모"]
ABSENT  = ["지게차", "용접 불꽃이 튀는 작업", "자동차 엔진", "바다 위의 배"]
rows = []
for kind, qs in [("있음", PRESENT), ("없음", ABSENT)]:
    for q in qs:
        s = (texts([q]) @ ALL.T)[0]; top = s.topk(3)
        rows.append((kind, q, round(float(top.values[0]), 3), ", ".join(meta.file[i] for i in top.indices.tolist())))
r = pd.DataFrame(rows, columns=["150장에", "질의", "1위 코사인", "상위 3장"])
print(r.to_string(index=False))
lo_present, hi_absent = r[r["150장에"] == "있음"]["1위 코사인"].min(), r[r["150장에"] == "없음"]["1위 코사인"].max()
print(f"\\n있는 것의 1위 최저 {lo_present:.3f} · 없는 것의 1위 최고 {hi_absent:.3f} → "
      + ("기준선 하나로 가를 수 있음(이 질의들에서는)" if lo_present > hi_absent else "겹침 — 기준선 하나로는 못 가름 → 리랭커 · VLM 확인(03 셀 10)"))
''')

md('''
### 셀 15 · 내 사진 · 내 문장으로
셀 1 의 `MY_IMAGE` · `MY_TEXTS` · `MY_INSTRUCTION` 을 바꿔 다시 실행한다. 한국어 문장도 된다.
''')
code('''
# 셀 15 · 내 사진 · 내 문장으로
v = images([MY_IMAGE], MY_INSTRUCTION)[0]
s = texts(MY_TEXTS, MY_INSTRUCTION) @ v
show([MY_IMAGE], [MY_IMAGE], cols=1, size=2.6)
print(pd.DataFrame({"문장": MY_TEXTS, "코사인": s.numpy().round(4)}).to_string(index=False))
''')

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT, len(cells), 'cells')
