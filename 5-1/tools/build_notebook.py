"""5-1 실습 노트북 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook.py        → ../notebooks/01_CLIP_실습.ipynb (출력 없이)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / '01_CLIP_실습.ipynb'
cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': s.strip('\n').splitlines(keepends=True)})


md('''
# 실습 · CLIP — 사진과 글을 한 공간에서 재기

강의 「CLIP 대조학습」 · 「CLIP의 활용과 한계」 · 「이미지·텍스트 임베딩 파이프라인」 장표를 직접 돌려 본다.
미리 준비한 사진 150장(`5-1/images`)과 문장으로 CLIP ViT-B/32 를 구동해, 장표의 숫자가 실제로 나오는지 확인한다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | 모델 불러오기 · 사진 목록 | – |
| 3 | 이미지 타워 — 텐서 모양 따라가기 | 이미지 타워 · 텐서의 흐름 |
| 4 | 텍스트 타워 — 토큰 번호와 [EOS] | 텍스트 타워 · 텐서의 흐름 |
| 5 | 사진 150장 임베딩 | – |
| 6 | N×N 채점표와 손실 | 정렬·손실 — N×N 채점표 계산 |
| 7 | 제로샷 분류 — 후보 = 클래스 이름 문장 | 활용과 한계 2번 |
| 8 | 문구 앙상블 | 활용과 한계 2번 |
| 9 | 검색 — 후보 = 사진 150장 | 활용과 한계 1번 |
| 10 | 한계 ① 관계 · 결합 | 활용과 한계 3번 |
| 11 | 한계 ② 부정 — 안전모 미착용 | 활용과 한계 3번 |
| 12 | 한계 ③ 개수 | 활용과 한계 3번 |
| 13 | 내 사진 · 내 문장으로 | – |

- 실행 : **Run → Run All Cells** (첫 실행은 모델 약 600MB 를 받는다. 이후 전체 약 30초)
- 장치 : 셀 1 의 `DEVICE` — `auto`(있으면 NVIDIA GPU → 맥 Metal → CPU 순) · `cuda` · `mps` · `cpu`
- **PyTorch 가 든 Jupyter 에서 연다** : 5-1 도커(http://localhost:8889) 또는 직접 설치(5-1/README.md). 4-1 의 8888 에는 PyTorch 가 없다
- 사진은 `5-1/images/images.csv` 에 출처 · 라이선스와 함께 적혀 있다
''')

md('''
### 셀 1 · 설정
- `DEVICE` : 계산 장치. `auto` 는 NVIDIA GPU(`cuda`) → 맥 Metal(`mps`) → `cpu` 순으로 있는 것을 고름
  - 도커 CPU 컨테이너에서는 `cpu` 만, 도커 GPU 컨테이너 · 직접 설치에서는 `cuda` · `mps` 도 된다
  - 장치를 바꿔도 결과 숫자는 같다(소수 넷째 자리까지). 속도만 다르다
- `MODEL` : CLIP ViT-B/32 (이미지 224×224, 조각 32×32, 출력 512차원) — 장표와 같은 모델
- `IMAGE_DIR` : `auto` 는 이 노트북 옆의 `../images`
- `BREEDS_37` · `TEMPLATES` : 셀 7 · 8 제로샷 후보와 문구
- `MY_TEXTS` · `MY_IMAGE` : 셀 13 에서 내 문장과 내 사진으로 바꿔 본다
''')
code('''
# 셀 1 · 설정
DEVICE     = "auto"      # "auto" · "cuda"(NVIDIA GPU) · "mps"(맥 Metal) · "cpu"
MODEL      = "openai/clip-vit-base-patch32"
IMAGE_DIR  = "auto"      # auto = 이 노트북 옆의 ../images

# 셀 7 · 8 : 제로샷 분류 후보 = Oxford-IIIT Pet 37품종 전부 (사진은 그중 8품종 × 5장)
BREEDS_37 = ["abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle", "bengal",
             "birman", "bombay", "boxer", "british_shorthair", "chihuahua", "egyptian_mau", "english_cocker_spaniel",
             "english_setter", "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin", "keeshond",
             "leonberger", "maine_coon", "miniature_pinscher", "newfoundland", "persian", "pomeranian", "pug",
             "ragdoll", "russian_blue", "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu", "siamese",
             "sphynx", "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"]
TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a photo of the {}, a type of pet.",
             "a blurry photo of a {}.", "a cropped photo of a {}.", "a good photo of a {}.",
             "a photo of my {}.", "a low resolution photo of a {}."]

# 셀 13 : 내 문장 · 내 사진 (images 폴더 파일 이름 또는 /work 아래 경로)
MY_TEXTS = ["a construction worker wearing a hard hat", "people in suits at a meeting", "a dog on the grass"]
MY_IMAGE = "hh_on_01.jpg"
''')

md('''
### 셀 2 · 모델 불러오기 · 사진 목록
- 셀 1 의 `DEVICE` 대로 장치를 고르고, 모델을 그 장치에 올린다
- PyTorch 가 없는 Jupyter 에서 열면 여기서 멈추고 어디서 열어야 하는지 알려 준다
- `CLIPModel` 하나에 이미지 타워 · 텍스트 타워 · 투영 · 온도가 다 들어 있다
- `images.csv` : 파일 · 그룹 · 라벨 · 출처 · 라이선스
''')
code('''
# 셀 2 · 모델 불러오기 · 사진 목록
import math, os, time, warnings
from pathlib import Path
# 모델 내려받기 · 불러오기 때 나오는 안내 문구와 진행 막대를 숨김(도커 이미지는 환경변수로 이미 숨김)
for k, v in {"HF_HUB_VERBOSITY": "error", "TRANSFORMERS_VERBOSITY": "error",
             "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}.items():
    os.environ.setdefault(k, v)
warnings.filterwarnings("ignore", message="IProgress not found")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"'{e.name}' 가 없는 Jupyter 다. 이 노트북은 PyTorch 가 든 환경에서 연다\\n"
        "  · 도커 : 5-1 폴더에서 docker compose up -d --build → http://localhost:8889  (4-1 의 8888 이 아님)\\n"
        "  · 도커 없이(맥 Metal · NVIDIA GPU) : 5-1/README.md 의 '직접 설치'") from None

# 한글 글꼴 : 컨테이너 NanumGothic · 맥 AppleGothic · 윈도 Malgun Gothic 중 있는 것
have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "AppleGothic", "Malgun Gothic") if f in have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 110})
torch.set_grad_enabled(False)

# 장치 고르기 — GPU 에서도 CPU 와 같은 숫자가 나오게 TF32(저정밀 곱셈)는 끈다
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
avail = {"cuda": torch.cuda.is_available(),
         "mps": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
         "cpu": True}
want = next(d for d in ("cuda", "mps", "cpu") if avail[d]) if DEVICE == "auto" else DEVICE
if not avail.get(want, False):
    raise RuntimeError(f"DEVICE = '{DEVICE}' 를 이 환경에서 쓸 수 없다. 쓸 수 있는 장치 : {[d for d, ok in avail.items() if ok]}"
                       " → 셀 1 을 바꾸거나 5-1/README.md 의 실행 환경 표를 본다")
DEV = torch.device(want)
name = torch.cuda.get_device_name(DEV) if DEV.type == "cuda" else ("Apple GPU (Metal)" if DEV.type == "mps" else "CPU")
print(f"장치 {DEV} · {name} · torch {torch.__version__}")

if IMAGE_DIR == "auto":
    IMAGE_DIR = next((str(d) for d in (Path.cwd().parent / "images", Path("/work/5-1/images"), Path.cwd() / "5-1" / "images")
                      if (d / "images.csv").exists()), None)
    assert IMAGE_DIR, "사진 폴더(5-1/images)를 찾지 못했다 → 셀 1 의 IMAGE_DIR 에 경로를 적는다"

t0 = time.time()
model = CLIPModel.from_pretrained(MODEL).eval().to(DEV)
proc  = CLIPProcessor.from_pretrained(MODEL)
print(f"모델 불러오기 {time.time() - t0:.1f}s · 파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

meta = pd.read_csv(Path(IMAGE_DIR) / "images.csv")
print(meta.groupby("group").size().to_string())

def feats(out):
    """get_image_features · get_text_features 결과를 [N × 512] 텐서로
    (transformers 5 는 출력 객체의 pooler_output 에, 4 는 텐서로 바로 준다)"""
    return out.pooler_output if hasattr(out, "pooler_output") else out

def load(name):
    p = Path(name) if "/" in str(name) else Path(IMAGE_DIR) / name
    return Image.open(p).convert("RGB")

def show(names, titles=None, cols=8, size=1.6):
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * (size + 0.35)))
    for ax in np.array(axes).reshape(-1):
        ax.axis("off")
    for i, n in enumerate(names):
        ax = np.array(axes).reshape(-1)[i]
        ax.imshow(load(n)); ax.set_title(titles[i] if titles else n, fontsize=7)
    plt.tight_layout(); plt.show()
''')

md('''
### 셀 3 · 이미지 타워 — 텐서 모양 따라가기
장표 「이미지 타워 — 텐서의 흐름」의 ②~⑦을 실제 모델에 걸어 모양을 찍는다.
- ② 입력 `[1, 3, 224, 224]` → ③ 조각내서 벡터로 `[1, 768, 7, 7]` = 조각 49개 × 768
- ④ [CLS] 붙이고 ⑤ 위치 더하면 `[1, 50, 768]` → ⑥ 블록 12개를 지나도 모양 그대로
- ⑦ [CLS] 행만 꺼내 `[1, 768]` → 투영 `[1, 512]` → 길이 1
- ③의 W 는 코드에서 `Conv2d(3, 768, kernel_size=32, stride=32)` 한 줄 — 장표의 "코드에선 Conv2d 한 줄"
''')
code('''
# 셀 3 · 이미지 타워 — 텐서 모양 따라가기
vm = model.vision_model
img = load("pet_shiba_inu_4.jpg")
x = proc(images=img, return_tensors="pt")["pixel_values"].to(DEV)
print(f"② 입력                     {tuple(x.shape)}   숫자 {x.numel():,}개")

conv = vm.embeddings.patch_embedding
print(f"   W = {conv}")
print(f"   W 모양 {tuple(conv.weight.shape)} = 768 × (3×32×32 = {3 * 32 * 32})")
patches = conv(x)
print(f"③ 조각내서 벡터로 (Conv2d)   {tuple(patches.shape)}   = 조각 7×7 = 49개 × 768")

h = vm.embeddings(x)
print(f"④⑤ [CLS] + 위치 임베딩       {tuple(h.shape)}")

h = vm.pre_layrnorm(h)
for i, layer in enumerate(vm.encoder.layers):
    out = layer(h, attention_mask=None, causal_attention_mask=None)
    h = out[0] if isinstance(out, tuple) else out
    if i in (0, 11):
        print(f"⑥ 블록 {i + 1:>2} 통과              {tuple(h.shape)}")

cls = vm.post_layernorm(h[:, 0, :])
print(f"⑦ [CLS] 행만                 {tuple(cls.shape)}")
proj = model.visual_projection(cls)
print(f"   투영 W_proj {tuple(model.visual_projection.weight.shape[::-1])}  {tuple(proj.shape)}")
emb = proj / proj.norm(dim=-1, keepdim=True)
print(f"   L2 정규화 → 길이 {emb.norm().item():.4f}")

ref = feats(model.get_image_features(pixel_values=x))
ref = ref / ref.norm(dim=-1, keepdim=True)
print(f"\\n직접 따라간 결과 = 모델 한 줄 결과 ? 최대 차이 {(emb - ref).abs().max().item():.2e}")
''')

md('''
### 셀 4 · 텍스트 타워 — 토큰 번호와 [EOS]
장표 「텍스트 타워」 2번의 토큰 번호가 실제로 나오는지, 그리고 대표 벡터가 [EOS] 자리에서 나오는지 확인한다.
- `"a red car"` → `[SOS] a red car [EOS]` = 49406 · 320 · 736 · 1615 · 49407
- 인과 마스크 : 앞 토큰은 뒤를 못 본다 → 문장 전체를 본 것은 [EOS] 하나
- 한글 문장은 잘게 쪼개진다 (학습이 영어 캡션 위주)
''')
code('''
# 셀 4 · 텍스트 타워 — 토큰 번호와 [EOS]
tok = proc.tokenizer
ids = tok("a red car")["input_ids"]
print(f"'a red car'   → 토큰 {len(ids)}개 {ids}")
print(" " * 16, tok.convert_ids_to_tokens(ids))
ko = "빨간 자동차"
ids = tok(ko)["input_ids"]
print(f"\\n'{ko}' → 토큰 {len(ids)}개 {ids}")
print(f"  한글 {len(ko.replace(' ', ''))}글자가 [SOS]·[EOS] 빼고 {len(ids) - 2}조각 — 글자를 UTF-8 바이트 조각으로 쪼갬(영어 위주 어휘)")

t = tok(["a red car"], padding="max_length", max_length=77, return_tensors="pt").to(DEV)
print(f"\\n77칸으로 채운 입력 {tuple(t['input_ids'].shape)} · 실제 토큰 {int(t['attention_mask'].sum())}개")
out = model.text_model(**t)
print(f"블록 출력 {tuple(out.last_hidden_state.shape)}")
eos = int(t["input_ids"][0].tolist().index(tok.eos_token_id))
same = torch.allclose(out.pooler_output[0], out.last_hidden_state[0, eos], atol=1e-5)
print(f"[EOS] 위치 = {eos}번째 · 대표 벡터가 [EOS] 행과 같은가 : {same}")
print(f"투영 W_text {tuple(model.text_projection.weight.shape[::-1])} → [512]")
''')

md('''
### 셀 5 · 사진 150장 임베딩
- 이미지 타워 한 번에 16장씩 → `[150 × 512]`, 모두 길이 1. 계산은 `DEV` 에서, 결과는 CPU 로 가져와 이후 셀에서 씀
- `enc_text(문장들)` : 텍스트 타워 → `[문장 수 × 512]`, 길이 1
''')
code('''
# 셀 5 · 사진 150장 임베딩
def enc_images(names, bs=16):
    out = []
    for i in range(0, len(names), bs):
        px = proc(images=[load(n) for n in names[i:i + bs]], return_tensors="pt")["pixel_values"].to(DEV)
        out.append(feats(model.get_image_features(pixel_values=px)).cpu())
    e = torch.cat(out)
    return e / e.norm(dim=-1, keepdim=True)

def enc_text(texts):
    t = proc(text=list(texts), return_tensors="pt", padding=True).to(DEV)
    e = feats(model.get_text_features(**t)).cpu()
    return e / e.norm(dim=-1, keepdim=True)

t0 = time.time()
IMG = enc_images(meta["file"].tolist())
print(f"사진 {IMG.shape[0]}장 → {tuple(IMG.shape)} · {DEV} {time.time() - t0:.1f}s · 길이 {IMG.norm(dim=-1).min():.4f}~{IMG.norm(dim=-1).max():.4f}")
emb = dict(zip(meta["file"], IMG))
''')

md('''
### 셀 6 · N×N 채점표와 손실
장표 「정렬·손실」을 실제 사진 4장과 캡션 4개로 계산한다.
- `S = I · Tᵀ` : 칸 하나 = 코사인. 대각선 = 자기 짝
- 배율 `exp(logit_scale)` 을 곱해 softmax → 행 방향 · 열 방향 cross-entropy
- 장표의 예시는 학습 **초기** 배율 ×14.3. 학습이 **끝난** 모델은 ×100 — 두 배율로 모두 계산한다
- 코사인 값 자체는 0.3 안팎으로 낮다. 이미지와 글은 한 공간에 있어도 서로 떨어져 모여 있다 → 값보다 **순위**로 판단(3강)
''')
code('''
# 셀 6 · N×N 채점표와 손실
pairs = [("pet_shiba_inu_4.jpg", "a photo of a Shiba Inu"),
         ("pet_persian_2.jpg",   "a photo of a Persian cat"),
         ("hh_on_02.jpg",        "a construction worker wearing a hard hat"),
         ("shape_above_red_01.png", "a red square above a blue square")]
I = torch.stack([emb[f] for f, _ in pairs]); T = enc_text([c for _, c in pairs])
S = I @ T.T
scale = model.logit_scale.exp().item()
print(f"학습된 배율 exp(logit_scale) = {scale:.1f}  (시작값 {1 / 0.07:.1f}, 상한 100)\\n")
print(pd.DataFrame(S.numpy(), index=[f"사진 {i + 1}" for i in range(4)], columns=[f"캡션 {i + 1}" for i in range(4)]).round(3))

for name, k in [("학습 초기 ×14.3", 1 / 0.07), (f"학습된 모델 ×{scale:.0f}", scale)]:
    logits = S * k
    p_row = logits.softmax(dim=1); p_col = logits.softmax(dim=0)
    loss_row = -p_row.diag().log(); loss_col = -p_col.diag().log()
    total = (loss_row.mean() + loss_col.mean()) / 2
    ce = (torch.nn.functional.cross_entropy(logits, torch.arange(4)) + torch.nn.functional.cross_entropy(logits.T, torch.arange(4))) / 2
    print(f"\\n[{name}]")
    print("  행 방향 정답 확률", p_row.diag().numpy().round(3), "· 손실", loss_row.numpy().round(3))
    print("  열 방향 정답 확률", p_col.diag().numpy().round(3), "· 손실", loss_col.numpy().round(3))
    print(f"  전체 손실 = (행 평균 + 열 평균) ÷ 2 = {total:.4f}  · 파이토치 cross_entropy 로 = {ce:.4f}")
print("\\n주제가 서로 다른 4쌍이라 쉬운 문제 → 학습이 끝난 배율에서는 손실이 거의 0. 실제 학습은 배치 3만 2천 쌍 속 헷갈리는 짝을 가른다")

fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.imshow(S.numpy(), cmap="Oranges"); ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels([f"캡션 {i + 1}" for i in range(4)]); ax.set_yticklabels([f"사진 {i + 1}" for i in range(4)])
for r in range(4):
    for c in range(4):
        ax.text(c, r, f"{S[r, c]:.2f}", ha="center", va="center", fontsize=9)
ax.set_title("S : 행 = 사진, 열 = 캡션 (대각선 = 자기 짝)", fontsize=9); plt.tight_layout(); plt.show()
''')

md('''
### 셀 7 · 제로샷 분류 — 후보 = 클래스 이름 문장
장표 「활용과 한계」 2번. 분류기를 따로 학습하지 않는다.
- 후보 : Oxford-IIIT Pet 의 **37품종 이름 전부**를 문장으로 → 텍스트 타워 → `[37 × 512]` = **분류 가중치 자리**
- 사진 40장(그중 8품종) `[40 × 512]` × 후보ᵀ `[512 × 37]` = `[40 × 37]` 점수 → 행마다 최고점이 답
- 문장만 바꾸면 분류기가 바뀐다 : 끝에서 같은 사진을 "개 / 고양이" 두 칸짜리 분류기로
''')
code('''
# 셀 7 · 제로샷 분류
pets = meta[meta.group == "pet"].reset_index(drop=True)
P = torch.stack([emb[f] for f in pets.file])
NAMES = [n.replace("_", " ") for n in BREEDS_37]

def zeroshot(W, labels, truth):
    score = P @ W.T
    pred = [labels[i] for i in score.argmax(dim=1).tolist()]
    return pred, float(np.mean([p == t for p, t in zip(pred, truth)])), score

W1 = enc_text([f"a photo of a {n}." for n in NAMES])
pred, acc, score = zeroshot(W1, BREEDS_37, pets.label.tolist())
print(f"분류 가중치 = 후보 문장 벡터 {tuple(W1.shape)} · 사진 {tuple(P.shape)} → 점수 {tuple(score.shape)}")
print(f"37개 후보 중에서 맞힌 비율 {acc:.3f}  ({sum(p == t for p, t in zip(pred, pets.label))}/{len(pets)})")
wrong = [(f, t, p) for f, t, p in zip(pets.file, pets.label, pred) if t != p]
print("틀린 사진 :", ", ".join(f"{t}→{p}" for _, t, p in wrong))
if wrong:
    show([f for f, _, _ in wrong], [f"{t}\\n→ {p}" for _, t, p in wrong], cols=min(6, len(wrong)), size=1.8)

truth2 = ["cat" if k in ("siamese", "persian", "bengal") else "dog" for k in pets.label]
_, acc2, _ = zeroshot(enc_text(["a photo of a dog", "a photo of a cat"]), ["dog", "cat"], truth2)
print(f"같은 사진, 후보만 두 문장으로 → 개/고양이 분류기 {acc2:.3f}")
''')

md('''
### 셀 8 · 문구 앙상블
장표 「활용과 한계」 2번 마지막 줄 — "문구가 정확도를 좌우 → 템플릿 선택 · 여러 문장 앙상블".
- 같은 사진 · 같은 37후보에서 **문구만** 바꾼다
- 앙상블 = 한 품종의 문장 여러 개를 텍스트 타워에 넣어 **벡터를 평균** → 후보 하나로 씀. 사진을 여러 번 보는 게 아님
- 정답 확률 : 학습된 배율(×100)로 softmax 했을 때 정답 품종이 받은 확률의 평균
''')
code('''
# 셀 8 · 문구 앙상블
def ensemble(templates):
    W = torch.stack([enc_text([t.format(n) for t in templates]).mean(dim=0) for n in NAMES])
    return W / W.norm(dim=-1, keepdim=True)

scale = model.logit_scale.exp().item()
rows = []
for name, W in [("라벨 단어만", enc_text(NAMES)),
                ("a photo of a {}.", W1),
                ("a photo of a {}, a type of pet.", enc_text([f"a photo of a {n}, a type of pet." for n in NAMES])),
                (f"문장 {len(TEMPLATES)}개 평균(앙상블)", ensemble(TEMPLATES))]:
    pred, acc, score = zeroshot(W, BREEDS_37, pets.label.tolist())
    prob = (score * scale).softmax(dim=1)[range(len(pets)), [BREEDS_37.index(l) for l in pets.label]].mean().item()
    rows.append((name, acc, prob))
print(pd.DataFrame(rows, columns=["후보 문장", "맞힌 비율", "정답 확률 평균"]).round(3).to_string(index=False))
print("\\n사진 40장이라 한 장 = 0.025. CLIP 논문(ImageNet) : 문장으로 감싸기 +1.3%p, 앙상블 80개 +3.5%p")
''')

md('''
### 셀 9 · 검색 — 후보 = 사진 150장
장표 「활용과 한계」 1번 — 같은 거리 재기인데 후보를 사진으로 두면 검색.
- 문장 하나 `[1 × 512]` × 사진 150장ᵀ → 점수 150개 → 상위 5장
''')
code('''
# 셀 9 · 검색
ALL = torch.stack([emb[f] for f in meta.file])
for q in ["a fluffy white dog", "a worker in an orange safety vest", "a cat with spotted fur"]:
    s = (enc_text([q]) @ ALL.T)[0]
    top = s.topk(5)
    names = [meta.file[i] for i in top.indices.tolist()]
    print(f"{q!r}  상위 5 코사인 {top.values.numpy().round(3)}")
    show(names, [f"{n}\\n{v:.3f}" for n, v in zip(names, top.values.tolist())], cols=5, size=1.9)
''')

md('''
### 셀 10 · 한계 ① 관계 · 결합
장표 「활용과 한계」 3번 — "문장 순서 · 관계 무시, 들어있는 단어들로만 봄".
- 관계 : 빨강 네모가 위 / 파랑 네모가 위 (각 10장) — 두 문장은 **단어가 같고 순서만** 다르다
- 결합 : 빨강 원 + 파랑 네모 / 파랑 원 + 빨강 네모 (각 10장) — 색이 어느 모양에 붙었는지
- 사진마다 두 문장 중 가까운 쪽을 고른다. 찍기면 0.5
- 대조군 : 단어가 **다른** 두 문장("흰 바탕의 색 도형" vs "개 사진")은 가르는가
''')
code('''
# 셀 10 · 한계 ① 관계 · 결합
def two_way(group, texts, first_label):
    g = meta[meta.group == group].reset_index(drop=True)
    X = torch.stack([emb[f] for f in g.file]); s = X @ enc_text(texts).T
    pick = s.argmax(dim=1).numpy()
    acc = float(((pick == 0) == (g.label == first_label)).mean())
    table = pd.crosstab(g.label.rename("그림"), pd.Series([texts[i] for i in pick], name="고른 문장"))
    ctrl = float(((X @ enc_text(["two colored shapes on a white background", "a photo of a dog"]).T).argmax(dim=1) == 0).float().mean())
    return acc, table, float((s[:, 0] - s[:, 1]).abs().mean()), ctrl

for title, group, texts, first in [
        ("관계(위/아래)", "shape_relation", ["a red square above a blue square", "a blue square above a red square"], "red above blue"),
        ("결합(색-모양)", "shape_binding", ["a red circle and a blue square", "a blue circle and a red square"], "red circle and blue square")]:
    acc, table, gap, ctrl = two_way(group, texts, first)
    print(f"■ {title}  맞힌 비율 {acc:.2f} (찍기 0.50) · 두 문장 코사인 차 평균 {gap:.4f} · 대조군 {ctrl:.2f}")
    print(table.to_string(), "\\n")
show(meta[meta.group == "shape_relation"].file.tolist()[::5] + meta[meta.group == "shape_binding"].file.tolist()[::5], cols=8, size=1.4)
''')

md('''
### 셀 11 · 한계 ② 부정 — 안전모 미착용
장표 「활용과 한계」 3번 — "'안전모 미착용 작업자'에 착용한 사진이 올라옴(부정 무시)".
- 현장 사진 40장 : 전원 미착용 20 · 전원 착용 20 (두 쪽 모두 같은 현장 사진 출처)
- 부정 문장으로 검색했을 때 상위 10장 중 **실제 미착용** 사진이 몇 장인가
- 비교 : 부정 없이 "안전모 쓴" 문장으로 검색하면 착용 사진을 잘 모으는가
''')
code('''
# 셀 11 · 한계 ② 부정
hh = meta[meta.group == "hardhat"].reset_index(drop=True)
H = torch.stack([emb[f] for f in hh.file])
def top_share(q, label, k=10):
    s = (enc_text([q]) @ H.T)[0]
    idx = s.topk(k).indices.tolist()
    return (hh.label[idx] == label).mean(), [hh.file[i] for i in idx]

rows = []
for q, want in [("a construction worker wearing a hard hat", "hardhat"),
                ("a construction worker without a hard hat", "no-hardhat"),
                ("a construction worker not wearing a hard hat", "no-hardhat"),
                ("people at a construction site with bare heads", "no-hardhat")]:
    share, top = top_share(q, want)
    rows.append((q, "착용" if want == "hardhat" else "미착용", share))
print(pd.DataFrame(rows, columns=["검색 문장", "찾으려는 사진", "상위 10 중 맞는 비율"]).to_string(index=False))
print("\\n전체 40장 중 절반이 미착용 → 무작위로 뽑아도 0.5")

share, top = top_share("a construction worker without a hard hat", "no-hardhat")
show(top, [f"{n}\\n{'미착용' if n.startswith('hh_off') else '착용'}" for n in top], cols=10, size=1.5)
''')

md('''
### 셀 12 · 한계 ③ 개수
- 검은 원 2~6개를 그린 그림 30장. 후보 문장 5개 `"a picture of N black circles"` 로 제로샷 분류
- 찍기 기준 0.2
''')
code('''
# 셀 12 · 한계 ③ 개수
cnt = meta[meta.group == "shape_count"].reset_index(drop=True)
C = torch.stack([emb[f] for f in cnt.file])
words = ["two", "three", "four", "five", "six"]
W = enc_text([f"a picture of {w} black circles" for w in words])
pred = [2 + i for i in (C @ W.T).argmax(dim=1).tolist()]
truth = cnt.label.astype(int).tolist()
print(f"개수 맞힘 {np.mean([p == t for p, t in zip(pred, truth)]):.2f}  (찍기 0.20)")
print(pd.crosstab(pd.Series(truth, name="실제"), pd.Series(pred, name="예측")))
''')

md('''
### 셀 13 · 내 사진 · 내 문장으로
셀 1 의 `MY_IMAGE` · `MY_TEXTS` 를 바꿔 다시 실행한다.
- 내 사진을 쓰려면 저장소 `5-1/images` 에 넣고 파일 이름만 적는다
''')
code('''
# 셀 13 · 내 사진 · 내 문장으로
v = enc_images([MY_IMAGE])[0]
s = enc_text(MY_TEXTS) @ v
p = (s * model.logit_scale.exp().item()).softmax(dim=0)
show([MY_IMAGE], [MY_IMAGE], cols=1, size=2.6)
print(pd.DataFrame({"문장": MY_TEXTS, "코사인": s.numpy().round(4), "softmax 확률": p.numpy().round(4)}).to_string(index=False))
''')

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT, len(cells), 'cells')
