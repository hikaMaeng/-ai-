"""5-1 실습 노트북 03 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook_vlm.py   → ../notebooks/03_VLM_실습.ipynb (출력 없이)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / '03_VLM_실습.ipynb'
cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': s.strip('\n').splitlines(keepends=True)})


md('''
# 실습 · VLM — Qwen3-VL-2B-Instruct

장표의 VLM 섹션(「VLM 패치」~「VLM 현시점 한계」)을 직접 돌려 본다. 01 · 02 와 **같은 사진 150장 · 문서 6장**을 쓴다.
02 의 임베딩 모델은 VLM 몸체에서 **마지막 자리 벡터**를 꺼냈고, 여기서는 같은 몸체가 **다음 토큰을 만들어 글로 답한다**.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | 모델 불러오기 · 묻는 함수 | – |
| 3 | 이미지 토큰 수 = 비용 — 크기별 토큰 수와 계산 시간 | VLM 패치 |
| 4 | 자리표 — 입력열에서 자리표 칸이 이미지 벡터로 바뀌는 것 확인 | VLM의 핵심 구조 3번 |
| 5 | 텐서의 흐름 — 사진 → 비전 인코더 → 커넥터 → LLM → 로짓 | VLM 구조 — 텐서의 흐름 |
| 6 | 커넥터 들여다보기 — 크기 · 구조 | 커넥터 |
| 7 | VLM 이 하는 일 — 설명 · 문서 판독 · 정해진 형식으로 답하기 | VLM의 진화 2번 |
| 8 | CLIP 이 못 한 것 다시 — 부정 · 관계 · 결합 · 개수 | CLIP의 활용과 한계 3번 |
| 9 | 한계 — 없는 것을 말하기 · 유도 질문 · 수치 읽기 | VLM 현시점 한계 1번 |
| 10 | 리랭커와 답 — "예" 확률로 다시 채점하고 근거 문서로 답하기 | VL 임베딩 — 사용법 2번 |
| 11 | 설명문을 붙이면 — VLM 설명문 + 02 임베딩, 네 가지 색인 비교 | VL 임베딩 — 사용법 1번 |
| 12 | 내 사진 · 내 질문으로 | – |

- 실행 : **Run → Run All Cells**. 첫 실행은 모델 약 4.5GB 를 받는다(셀 11 은 02 의 임베딩 모델 약 4GB 도). 받은 뒤 CPU 에서 전체 약 20분
- 메모리 : 모델을 float32 로 올리면 약 9GB, 셀 11 에서 임베딩 모델을 더 올리면 약 18GB. 모자라면 셀 1 `DTYPE = "bfloat16"`
- 장치 · 설치는 01 · 02 와 같다 : 셀 1 `DEVICE`, PyTorch 가 없는 Jupyter(4-1 의 8888)면 셀 2 가 처음 한 번 설치
- 이 모델은 **2B**(작은 쪽). 한국어 답에 다른 언어 낱말이 섞이거나 수치를 조금 틀리게 읽는다 — 그 자체가 관찰 대상이다
- 답은 늘 같은 글이 나오도록 **탐욕 디코딩**(가장 높은 토큰만, 2강 temperature 0 과 같음)으로 만든다
''')

md('''
### 셀 1 · 설정
- `MODEL` : Qwen3-VL **2B** Instruct(작은 쪽. 4B · 8B · 32B 도 있음). 장표 「VLM 구조 — 텐서의 흐름」은 8B 기준
- `DOC_TOKENS` : 문서 사진 한 장의 이미지 토큰 상한. 원본(900×1100)은 약 950개 → 512 로 줄여 시간을 아낀다(셀 3 에서 차이를 잰다)
- `RUN_FUSION` : 셀 11 은 임베딩 모델을 하나 더 올린다. 메모리가 모자라면 `False`
- `MY_*` : 셀 12 에서 바꿔 본다
''')
code('''
# 셀 1 · 설정
DEVICE     = "auto"      # "auto" · "cuda"(NVIDIA GPU) · "mps"(맥 Metal) · "cpu"
DTYPE      = "float32"   # "float32" · "bfloat16"(메모리 절반)
MODEL      = "Qwen/Qwen3-VL-2B-Instruct"
EMB_MODEL  = "Qwen/Qwen3-VL-Embedding-2B"   # 셀 11 에서만 (02 와 같은 모델)
IMAGE_DIR  = "auto"      # auto = 이 노트북 옆의 ../images
DOC_DIR    = "auto"      # auto = 이 노트북 옆의 ../docs
DOC_TOKENS = 512
RUN_FUSION = True

# 셀 12 : 내 사진 · 내 질문 (images · docs 폴더 파일 이름 또는 /work 아래 경로)
MY_IMAGE = "hh_off_03.jpg"
MY_QUESTIONS = ["이 사진을 한 문장으로 설명해.", "안전모를 쓰지 않은 사람은 몇 명인가?"]
''')

md('''
### 셀 2 · 모델 불러오기 · 묻는 함수
- PyTorch · torchvision · transformers 가 없으면 이 Jupyter 에 설치한다(처음 한 번). 02 와 같은 순서
- 모델은 `Qwen3VLForConditionalGeneration` = 02 의 몸체(`Qwen3VLModel` : 비전 인코더 + 커넥터 + LLM) **+ 다음 토큰을 고르는 출력 행렬(lm_head)**
- `ask(사진, 질문)` : 대화 틀을 만들어 → 모델 → 답을 글로. 탐욕 디코딩
- `p_yes(사진, 질문)` : 답을 만들지 않고 **첫 토큰이 "Yes" 일 확률**만 본다 — 로짓에서 Yes 와 No 두 칸만 꺼내 softmax(1강). 리랭커가 점수를 내는 방법과 같다(셀 10)
''')
code('''
# 셀 2 · 모델 불러오기 · 묻는 함수
import math, os, re, time, json, warnings
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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
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
meta = pd.read_csv(Path(IMAGE_DIR) / "images.csv")
docs = pd.read_csv(Path(DOC_DIR) / "docs.csv")

t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL)
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=getattr(torch, DTYPE)).eval().to(DEV)
tok = proc.tokenizer
IMG_PAD = model.config.image_token_id
print(f"모델 불러오기 {time.time() - t0:.1f}s · 파라미터 {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

def path_of(name):
    if "/" in str(name) or "\\\\" in str(name):
        return Path(name)
    return Path(DOC_DIR if str(name).startswith("doc_") else IMAGE_DIR) / name

def size_for(tokens):
    """이미지 토큰 상한 → 프로세서 설정. 이미지 토큰 1개 = 32×32 픽셀(16픽셀 패치 2×2)"""
    return {"longest_edge": tokens * 32 * 32, "shortest_edge": 64 * 32 * 32} if tokens else None

def prepare(image, question, tokens=None):
    img = image if isinstance(image, Image.Image) else Image.open(path_of(image)).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = {"size": size_for(tokens)} if tokens else {}
    return proc(text=[text], images=[img], return_tensors="pt", **kw).to(DEV)

def ask(image, question, n=60, tokens=None):
    """사진 + 질문 → 답(글). 탐욕 디코딩, 최대 n 토큰"""
    inp = prepare(image, question, tokens)
    out = model.generate(**inp, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

YES = [tok.convert_tokens_to_ids(t) for t in ("Yes", "yes")]
NO  = [tok.convert_tokens_to_ids(t) for t in ("No", "no")]
def p_yes(image, question, tokens=None):
    """답을 만들지 않고, 다음 토큰 로짓에서 Yes · No 칸만 꺼내 비교 → Yes 확률"""
    lg = model(**prepare(image, question + " Answer yes or no.", tokens)).logits[0, -1].float()
    return torch.sigmoid(torch.logsumexp(lg[YES], 0) - torch.logsumexp(lg[NO], 0)).item()

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
### 셀 3 · 이미지 토큰 수 = 비용
장표 「VLM 패치」 핵심 — "이미지 토큰 수 = 입력 길이 = 비용". Qwen3-VL 은 사진을 정사각형으로 누르지 않고 원본 비율대로 자른다.
- 16픽셀 패치 → 비전 인코더 뒤 이웃 2×2 를 토큰 1개로 → 이미지 토큰 1개 = 32×32 픽셀
- 프로세서의 허용 범위(256² ~ 4096² 픽셀)를 토큰으로 바꾸면 64 ~ 16384개
- 문서 사진 한 장을 토큰 상한만 바꿔 넣고, 모델이 입력을 한 번 읽는 데 걸리는 시간(답을 만들기 전 단계)을 잰다
''')
code('''
# 셀 3 · 이미지 토큰 수 = 비용
lim = proc.image_processor.size
print(f"허용 범위 {int(lim['shortest_edge'] ** .5)}² ~ {int(lim['longest_edge'] ** .5)}² 픽셀 → 이미지 토큰 {lim['shortest_edge'] // 1024} ~ {lim['longest_edge'] // 1024}개\\n")
for f in ["hh_on_01.jpg", "pet_pug_1.jpg", docs.file[0]]:
    im = Image.open(path_of(f)); inp = prepare(f, "?")
    print(f"{f:28s} {im.width}×{im.height} → 이미지 토큰 {int((inp['input_ids'] == IMG_PAD).sum())}개")

rows = []
for t in [None, 1024, 512, 256]:
    inp = prepare(docs.file[0], "이 문서의 판정 결과는?", t)
    model(**inp)                                   # 첫 호출 준비 시간은 빼고
    t0 = time.time(); model(**inp); dt = time.time() - t0
    rows.append(("원본" if t is None else f"상한 {t}", int((inp["input_ids"] == IMG_PAD).sum()), inp["input_ids"].shape[1], f"{dt:.2f}s"))
print("\\n문서 사진(900×1100) 한 장")
print(pd.DataFrame(rows, columns=["설정", "이미지 토큰", "입력열 전체", "한 번 읽기"]).to_string(index=False))
''')

md('''
### 셀 4 · 자리표 — 자리표 칸이 이미지 벡터로 바뀐다
장표 「VLM의 핵심 구조」 3번과 「VLM 구조 — 텐서의 흐름」 ⑤를 확인한다. 장표의 예시와 같은 **640×480 사진 + "이 배관 사진에서 부식 부위는?"**
- 장표의 실측(Qwen3-VL 8B) : 입력열 321토큰 = 글 토큰 21 + `<|image_pad|>` 300 — 2B 도 전처리기가 같아 같은 수가 나와야 한다
- LLM 에 실제로 들어가는 벡터 줄 `[321 × 폭]` 을 가로채서
  - 글 토큰 자리 = 어휘표에서 꺼낸 행과 같은지
  - 자리표 자리 = 커넥터가 만든 이미지 벡터와 같은지 를 비교한다
''')
code('''
# 셀 4 · 자리표
img640 = Image.open(path_of("hh_on_01.jpg")).convert("RGB").resize((640, 480))   # 크기만 장표 예시와 같게
Q = "이 배관 사진에서 부식 부위는?"
inp = prepare(img640, Q)
ids = inp["input_ids"][0]
is_img = ids == IMG_PAD
print(f"입력열 {len(ids)}토큰 = 글 토큰 {int((~is_img).sum())} + 자리표 {int(is_img.sum())}")
print("모델에 들어가는 글 :\\n" + tok.decode(ids).replace("<|image_pad|>" * int(is_img.sum()), f"<|image_pad|> × {int(is_img.sum())}"))

cap = {}
h = model.model.language_model.register_forward_pre_hook(lambda m, a, kw: cap.update(e=kw["inputs_embeds"][0]), with_kwargs=True)
model(**inp); h.remove()
E = cap["e"]                                                         # LLM 이 받는 벡터 줄
table = model.get_input_embeddings()(ids)                            # 어휘표에서 그냥 꺼낸 행
vis = model.model.get_image_features(inp["pixel_values"], inp["image_grid_thw"]).pooler_output   # 커넥터 출력
vis = torch.cat(list(vis)) if isinstance(vis, (list, tuple)) else vis
print(f"\\nLLM 입력 {tuple(E.shape)}")
print(f"글 토큰 자리 vs 어휘표 행      최대 차이 {(E[~is_img] - table[~is_img]).abs().max():.1e}")
print(f"자리표 자리 vs 커넥터 출력 {tuple(vis.shape)}  최대 차이 {(E[is_img] - vis).abs().max():.1e}")
print(f"자리표 자리 vs 어휘표의 <|image_pad|> 행  최대 차이 {(E[is_img] - table[is_img]).abs().max():.2f}  ← 바뀌었다")
''')

md('''
### 셀 5 · 텐서의 흐름
장표 「VLM 구조 — 텐서의 흐름」 ①~⑥을 층마다 모양으로 찍는다. 장표는 8B, 여기는 2B — 단계와 규칙은 같고 폭 · 층 수만 다르다.
''')
code('''
# 셀 5 · 텐서의 흐름
shapes = {}
grab = lambda k: (lambda m, a, o: shapes.__setitem__(k, tuple((o[0] if isinstance(o, tuple) else getattr(o, "last_hidden_state", o)).shape)))
v = model.model.visual
hooks = [v.patch_embed.register_forward_hook(grab("patch")), v.blocks[-1].register_forward_hook(grab("vit")),
         v.merger.register_forward_hook(grab("merger")), model.model.language_model.register_forward_hook(grab("llm"))]
out = model(**inp)
for h in hooks:
    h.remove()
c, vc, tc = model.config, model.config.vision_config, model.config.text_config
print(f"① 사진                      640×480")
print(f"② 펼침                      {tuple(inp['pixel_values'].shape)}   조각마다 16×16×3×2 = {16 * 16 * 3 * 2}개 값")
print(f"③ 비전 인코더 입력 투영      {shapes['patch']}")
print(f"   ViT {vc.depth}층 통과          {shapes['vit']}   폭 {vc.hidden_size} 유지")
print(f"④ 커넥터 : 2×2 이어 붙임   ({shapes['vit'][0] // 4}, {vc.hidden_size * 4}) → 2층 MLP → {shapes['merger']}")
print(f"⑤ LLM 입력열               {tuple(E.shape)}")
print(f"⑥ LLM {tc.num_hidden_layers}층 통과            {shapes['llm']}   폭 {tc.hidden_size} 유지")
print(f"   × 출력 행렬 → 로짓        {tuple(out.logits.shape)}")
p = out.logits[0, -1].float().softmax(-1); top = p.topk(5)
print("\\n마지막 자리의 다음 토큰 후보 :", ", ".join(f"{tok.decode([i])!r} {pv:.3f}" for i, pv in zip(top.indices.tolist(), top.values.tolist())))
print("\\n장표(8B)와 비교")
print(pd.DataFrame([("비전 인코더", f"{vc.depth}층 · 폭 {vc.hidden_size}", "27층 · 폭 1152"),
                    ("커넥터", f"{vc.hidden_size * 4} → {vc.out_hidden_size}", "4608 → 4096"),
                    ("LLM", f"{tc.num_hidden_layers}층 · 폭 {tc.hidden_size}", "36층 · 폭 4096"),
                    ("어휘", f"{tc.vocab_size:,}", "151,936")], columns=["", "2B (여기)", "8B (장표)"]).to_string(index=False))
''')

md('''
### 셀 6 · 커넥터 들여다보기
장표 「커넥터」 1번 "폭 맞춤 · 의미 맞춤 · 개수 조절" 과 4번 DeepStack.
- `merger` : 이웃 2×2 패치 벡터 4개를 이어 붙여(개수 1/4) → LayerNorm → 선형 → GELU → 선형(폭을 LLM 입력 폭으로)
- `deepstack_merger_list` : ViT 중간층 특징을 같은 방식으로 바꿔 LLM 앞쪽 층에 더해 주는 커넥터들
- 부품별 파라미터 수 — 커넥터는 전체의 몇 % 인가
''')
code('''
# 셀 6 · 커넥터
print(v.merger)
count = lambda m: sum(p.numel() for p in m.parameters())
parts = [("비전 인코더(ViT)", count(v.patch_embed) + count(v.pos_embed) + count(v.blocks)),
         ("커넥터 merger", count(v.merger)),
         (f"DeepStack 커넥터 {len(v.deepstack_merger_list)}개 (ViT {list(vc.deepstack_visual_indexes)}층 → LLM 앞 {len(v.deepstack_merger_list)}개 층)", count(v.deepstack_merger_list)),
         ("LLM (어휘표 포함)", count(model.model.language_model))]
total = count(model)
print(pd.DataFrame([(n, f"{c / 1e6:,.1f}M", f"{c / total:.1%}") for n, c in parts], columns=["부품", "파라미터", "비율"]).to_string(index=False))
print(f"\\n출력 행렬(lm_head)은 어휘표와 같은 웨이트를 나눠 씀 : {model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()}")
''')

md('''
### 셀 7 · VLM 이 하는 일 — 설명 · 문서 판독 · 정해진 형식으로 답하기
장표 「VLM의 진화」 2번 "흡수된 작업" — 캡션 · 문서 판독 · 구조화 출력(JSON).
- 사진 설명 : 같은 사진을 영어 · 한국어로
- 문서 판독 : 표와 그래프가 있는 문서 사진에서 값을 읽어 답한다. 글자를 따로 뽑아내는(OCR) 단계 없음
- 정해진 형식 : 점검표처럼 항목을 정해 주고 JSON 으로만 답하게 한다 → 코드가 바로 읽을 수 있다(시리즈 원칙 "판단은 모델, 통제는 코드")
''')
code('''
# 셀 7 · VLM 이 하는 일
show(["hh_on_02.jpg", "hh_off_03.jpg", docs.file[0], docs.file[5]], ["hh_on_02", "hh_off_03", "문서 1", "문서 6"], cols=4, size=2.4)
t0 = time.time()
print("[설명] hh_on_02.jpg")
print("  영어   :", ask("hh_on_02.jpg", "Describe this photo in one sentence.", 40))
print("  한국어 :", ask("hh_on_02.jpg", "이 사진을 한국어 한 문장으로 설명해.", 40))
print("\\n[문서 판독]")
for f, q in [(docs.file[0], "이 문서에서 시험 압력과 판정 결과만 한 줄로 답해."),
             (docs.file[5], "그래프에서 온도가 가장 높았던 시각과 온도, 경보 원인을 한 줄로 답해."),
             (docs.file[3], "톨루엔을 다룰 때 필요한 호흡 보호구만 답해.")]:
    print(f"  {f} · {q}\\n    → {ask(f, q, 60, DOC_TOKENS)}")
print("\\n[정해진 형식] hh_off_03.jpg")
raw = ask("hh_off_03.jpg", '다음 JSON 형식으로만 답해: {"인원수": 숫자, "안전모_미착용_인원": 숫자, "장소": "짧은 말"}', 60)
print("  답 :", raw)
m = re.search(r"\\{.*\\}", raw, re.S)
try:
    print("  코드가 읽은 값 :", json.loads(m.group(0)) if m else "JSON 없음")
except json.JSONDecodeError as e:
    print("  JSON 읽기 실패 :", e)
print(f"\\n{time.time() - t0:.0f}s")
''')

md('''
### 셀 8 · CLIP 이 못 한 것 다시 — 부정 · 관계 · 결합 · 개수
장표 「CLIP의 활용과 한계」 3번 "→ VLM 필요" 를 01 · 02 와 같은 사진으로 잰다.
- VLM 은 사진마다 **질문에 답한다**(예/아니오 · 숫자). 01 · 02 는 문장과의 **거리**로 골랐다 — 같은 사진, 다른 방식
- 부정 : 안전모 사진 40장마다 "안전모를 안 쓴 사람이 있나?" → `p_yes` > 0.5 를 "있다" 로
- 관계 · 결합 : 도형 20장씩. 개수 : 검은 원 30장, 숫자 하나로 답하게
- 01 · 02 열은 그 노트북의 실측값(부정은 "상위 10장 중 미착용 비율" 이라 같은 잣대는 아님)
''')
code('''
# 셀 8 · CLIP 이 못 한 것 다시
t0 = time.time()
def judge(group, question, yes_label):
    g = meta[meta.group == group].reset_index(drop=True)
    p = np.array([p_yes(f, question) for f in g.file])
    return float(((p > 0.5) == (g.label == yes_label)).mean()), g, p

acc_neg, hh, p_neg = judge("hardhat", "Is there any person in this photo who is NOT wearing a hard hat?", "no-hardhat")
acc_rel, *_ = judge("shape_relation", "Is the red square above the blue square?", "red above blue")
acc_bind, *_ = judge("shape_binding", "Is the circle red and the square blue?", "red circle and blue square")
cnt = meta[meta.group == "shape_count"].reset_index(drop=True)
said = [ask(f, "How many black circles are in this image? Answer with a single number.", 3) for f in cnt.file]
pred = [int(m.group()) if (m := re.search(r"\\d+", s)) else -1 for s in said]
acc_cnt = float(np.mean([p == t for p, t in zip(pred, cnt.label.astype(int))]))

print(pd.DataFrame([("부정 : 안전모 미착용 (40장)", "0.00 *", "0.00 *", acc_neg),
                    ("관계 : 빨강/파랑 네모 위아래 (20장)", "0.40", "0.50", acc_rel),
                    ("결합 : 원 · 네모의 색 (20장)", "0.10", "0.45", acc_bind),
                    ("개수 : 검은 원 2~6개 (30장)", "0.37", "0.87", acc_cnt)],
                   columns=["시험", "01 CLIP", "02 VL 임베딩", "03 VLM"]).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("* 01 · 02 의 부정은 '안전모를 쓰지 않은 작업자' 로 검색한 상위 10장 중 미착용 비율")
print("\\n개수 — 실제 × 답"); print(pd.crosstab(pd.Series(cnt.label.astype(int).tolist(), name="실제"), pd.Series(pred, name="답")))
print(f"\\n{time.time() - t0:.0f}s · 사진 {len(hh) + 40 + len(cnt)}장")
''')

md('''
### 셀 9 · 한계 — 없는 것을 말하기 · 유도 질문 · 수치 읽기
장표 「VLM 현시점 한계」 1번 — 시각 환각 · 유도 질문에 끌려감 · 정밀 수치.
- 안전모 사진 40장에 "개가 있나?" — 40장에 개는 없다(강사가 직접 확인). "있다" 로 답한 사진이 있으면 띄운다
  - 지게차로 물으면 배경에 노란 건설 장비가 흐릿하게 보이는 2장(hh_off_02 · hh_on_11)에서 "있다" 가 나온다 — 사람도 가리기 어려워 환각 시험에서 뺐다
- 지게차가 없는 실내 사진(hh_on_02)에 "지게차는 무슨 색인가?" — 질문이 "있다" 를 전제로 깔고 있다
- 모두 안전모를 쓴 사진에 "몇 번째 사람이 안전모를 벗고 있나?"
- 그래프의 최고 온도 : 문서 6 의 실제 값은 189 °C (15:00)
''')
code('''
# 셀 9 · 한계
t0 = time.time()
p_dog = np.array([p_yes(f, "Is there a dog in this photo?") for f in hh.file])
said_yes = list(hh.file[p_dog > 0.5])
print(f"개가 없는 사진 {len(hh)}장 중 '있다'(p_yes > 0.5) : {len(said_yes)}장 {said_yes} · p_yes 최대 {p_dog.max():.3f}")
if said_yes:
    show(said_yes, [f"{f}\\np_yes {p:.3f}" for f, p in zip(said_yes, p_dog[p_dog > 0.5])], cols=max(2, len(said_yes)), size=2.4)
print("\\n[전제를 깐 질문] hh_on_02.jpg (지게차 없음 · 한 사람이 안전모를 씀)")
for q in ["사진 속 지게차는 무슨 색인가?", "이 사진에서 몇 번째 사람이 안전모를 벗고 있나?", "이 사진에 지게차가 있나? 있으면 색을, 없으면 없다고 답해."]:
    print(f"  {q}\\n    → {ask('hh_on_02.jpg', q, 50)}")
print("\\n[수치 읽기] 문서 6 의 최고 온도 (실제 189 °C, 15:00)")
print("  →", ask(docs.file[5], "그래프에서 가장 높은 온도 값과 그 시각만 답해.", 30, DOC_TOKENS))
print(f"\\n{time.time() - t0:.0f}s")
''')

md('''
### 셀 10 · 리랭커와 답 — 근거 문서로 답하기
장표 「VL 임베딩 — 사용법」 2번 : 임베딩으로 후보를 찾고(02 셀 10) → **리랭커가 질의와 후보를 함께 읽고 다시 채점** → VLM 이 선택된 페이지로 답한다.
- 리랭커의 점수 = "이 문서가 질문에 답하는가?" 에 대한 **Yes 확률** (`p_yes`). 공개 리랭커(Qwen3-VL-Reranker)도 같은 원리를 따로 학습시킨 모델
- 후보 = 문서 6장 전부(02 의 1단 검색이 남긴 후보라고 가정). 02 셀 10 에서 임베딩이 틀린 질의 "배관 수압 시험 결과" 를 포함 — 리랭커가 바로잡는지 본다
- 근거 문서가 틀리면 답도 그 문서 내용으로 나온다 → 답 옆에 근거 문서를 꼭 함께 낸다
- 1위 문서를 VLM 에 넣고 답 + 근거 문서를 함께 낸다
''')
code('''
# 셀 10 · 리랭커와 답
t0 = time.time()
RAG_Q = ["배관 수압 시험 결과", "용접하기 전에 받아야 하는 허가", "반응기 온도가 경보치를 넘은 기록"]
for q in RAG_Q:
    s = np.array([p_yes(f, f"Does this document answer the question: '{q}'?", DOC_TOKENS) for f in docs.file])
    order = s.argsort()[::-1]
    print(f"질의 : {q}")
    print("  리랭커 점수 :", ", ".join(f"{docs.title[i]} {s[i]:.3f}" for i in order[:3]), "…")
    best = docs.file[order[0]]
    print(f"  답 : {ask(best, q + ' — 이 문서의 내용만으로 한두 줄로 답해.', 60, DOC_TOKENS)}")
    print(f"  근거 : {best} ({docs.title[order[0]]})\\n")
print(f"{time.time() - t0:.0f}s")
''')

md('''
### 셀 11 · 설명문을 붙이면 — 네 가지 색인 비교
장표 「VL 임베딩 — 사용법」 1번 "사진마다 VLM 설명문 + 따로 저장해 결과를 합침".
1. VLM 이 문서마다 **점검표형 설명문**을 쓴다(항목을 정해 준다)
2. 02 의 임베딩 모델로 색인을 네 가지로 만든다
   - ① 사진만 : 문서 사진 벡터
   - ② 설명문만 : 설명문 글 벡터
   - ③ 한 입력으로 : 사진 + 설명문을 한 입력으로 넣은 벡터 1개
   - ④ 따로 찾아 합치기 : ①과 ②의 **순위**를 합침 — 순위 r 마다 1/(60 + r) 을 더함(4강 하이브리드와 같은 RRF)
3. 질의 12개(02 셀 10 의 6개 + 작은 글씨 속 내용을 묻는 6개)로 1위가 정답 문서인 비율을 잰다
- 후보가 6장뿐이고 질의도 12개라 한 개 = 0.08. 방향을 보는 시험이지 결론이 아니다
''')
code('''
# 셀 11 · 설명문을 붙이면
FUSION_Q = [("배관 수압 시험 결과", 0), ("용접하기 전에 받아야 하는 허가", 1), ("펌프 진동 점검 결과", 2),
            ("톨루엔 다룰 때 필요한 보호구", 3), ("이번 달 안전 교육 일정", 4), ("반응기 온도가 경보치를 넘은 기록", 5),
            ("냉각수 라인 P-1203", 0), ("화재 감시자를 두는 작업", 1), ("베어링 온도 58도", 2),
            ("인화점이 4도인 물질", 3), ("소방 훈련 날짜", 4), ("필터 막힘으로 냉각수 유량 저하", 5)]
if not RUN_FUSION:
    print("셀 1 RUN_FUSION = False → 건너뜀")
else:
    t0 = time.time()
    CAPTION_Q = "이 문서를 검색용으로 정리해. 형식: 문서 종류 / 설비·물질 / 핵심 수치 / 판정·결론. 한국어로 네 줄."
    captions = [ask(f, CAPTION_Q, 90, DOC_TOKENS) for f in docs.file]
    for f, c in zip(docs.file, captions):
        print(f"[{f}]\\n{c}\\n")
    print(f"설명문 {len(captions)}개 {time.time() - t0:.0f}s\\n")

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    t0 = time.time()
    eproc = AutoProcessor.from_pretrained(EMB_MODEL, padding_side="right")
    emodel = Qwen3VLModel.from_pretrained(EMB_MODEL, dtype=getattr(torch, DTYPE)).eval().to(DEV)
    INS_DEFAULT, INS_DOC = "Represent the user's input.", "Retrieve document images that answer the user's query."
    def fit(im, maxp=1800 * 32 * 32):                     # 02 셀 2 와 같은 크기 맞춤(32픽셀 배수)
        h, w = max(32, round(im.height / 32) * 32), max(32, round(im.width / 32) * 32)
        if h * w > maxp:
            b = math.sqrt(im.height * im.width / maxp); h, w = math.floor(im.height / b / 32) * 32, math.floor(im.width / b / 32) * 32
        return im.resize((w, h))
    def embed(items, ins):
        out = []
        for it in items:
            content, imgs = [], []
            if it.get("image"):
                content.append({"type": "image"}); imgs.append(fit(Image.open(path_of(it["image"])).convert("RGB")))
            if it.get("text"):
                content.append({"type": "text", "text": it["text"]})
            conv = [{"role": "system", "content": [{"type": "text", "text": ins}]}, {"role": "user", "content": content}]
            text = eproc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)
            inp = eproc(text=text, images=imgs or None, padding=True, do_resize=False, return_tensors="pt").to(DEV)
            out.append(F.normalize(emodel(**inp).last_hidden_state[0, -1].float(), dim=-1).cpu())
        return torch.stack(out)
    IMG = embed([{"image": f} for f in docs.file], INS_DEFAULT)
    CAP = embed([{"text": c} for c in captions], INS_DEFAULT)
    BOTH = embed([{"image": f, "text": c} for f, c in zip(docs.file, captions)], INS_DEFAULT)
    QV = embed([{"text": q} for q, _ in FUSION_Q], INS_DOC)
    print(f"임베딩 모델 · 색인 · 질의 {time.time() - t0:.0f}s\\n")

    def ranks(S):                                          # 행 = 질의, 값 = 문서별 순위(0 = 1위)
        return S.argsort(dim=1, descending=True).argsort(dim=1)
    R_img, R_cap, R_both = ranks(QV @ IMG.T), ranks(QV @ CAP.T), ranks(QV @ BOTH.T)
    RRF = 1 / (60 + R_img + 1) + 1 / (60 + R_cap + 1)
    methods = {"① 사진만": (QV @ IMG.T), "② 설명문만": (QV @ CAP.T), "③ 한 입력으로": (QV @ BOTH.T), "④ 따로 찾아 합치기": RRF}
    gold = torch.tensor([g for _, g in FUSION_Q])
    table = pd.DataFrame({k: [docs.file[i][:5] for i in S.argmax(dim=1).tolist()] for k, S in methods.items()})
    table.insert(0, "정답", [docs.file[g][:5] for g in gold.tolist()]); table.insert(0, "질의", [q for q, _ in FUSION_Q])
    print(table.to_string(index=False))
    print("\\n1위가 정답인 비율 :", {k: round(float((S.argmax(dim=1) == gold).float().mean()), 2) for k, S in methods.items()})
    print("  앞 6개(문서 주제) :", {k: round(float((S[:6].argmax(dim=1) == gold[:6]).float().mean()), 2) for k, S in methods.items()})
    print("  뒤 6개(작은 글씨) :", {k: round(float((S[6:].argmax(dim=1) == gold[6:]).float().mean()), 2) for k, S in methods.items()})
    del emodel
''')

md('''
### 셀 12 · 내 사진 · 내 질문으로
셀 1 의 `MY_IMAGE` · `MY_QUESTIONS` 를 바꿔 다시 실행한다. 문서 사진(`doc_...`)도 된다.
''')
code('''
# 셀 12 · 내 사진 · 내 질문으로
show([MY_IMAGE], [MY_IMAGE], cols=1, size=2.6)
for q in MY_QUESTIONS:
    t0 = time.time()
    print(f"{q}\\n  → {ask(MY_IMAGE, q, 80, DOC_TOKENS if str(MY_IMAGE).startswith('doc_') else None)}  ({time.time() - t0:.0f}s)")
''')

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT, len(cells), 'cells')
