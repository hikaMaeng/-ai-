"""5-2 실습 노트북 02 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook_02.py   → ../notebooks/02_UNet과_프롬프트.ipynb (출력 없이)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / '02_UNet과_프롬프트.ipynb'
cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': s.strip('\n').splitlines(keepends=True)})


md('''
# 실습 02 · U-Net과 프롬프트 — 디노이저 안을 열어 보기

장표 「U-Net denoiser」~「U-Net과 타임스텝」, 신설 「교차 어텐션」 · 「CFG」를 SD1.5 의 U-Net 안에서 확인한다. 01 과 같은 모델 · 같은 시드.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | 모델 불러오기 · 공통 함수(01 셀 2 와 같음) | – |
| 3 | U-Net 층별 모양 — 해상도가 줄었다 느는 U 자, skip 으로 넘기는 텐서 | U-Net denoiser 3 · 4번 · U-Net denoiser — skip과 학습 2 · 3번 |
| 4 | skip 을 흐리게 하면 — skip 이 나르는 세부는 무엇인가 | U-Net denoiser — skip과 학습 4번 |
| 5 | 교차 어텐션의 크기 — 질의는 그림 칸, 키 · 값은 글 토큰 | 교차 어텐션 — 프롬프트가 들어가는 곳 · 어텐션 블록과 텐서 모양 |
| 6 | 단어별 어텐션 지도 — red · car · beach 가 그림의 어디를 보나 | U-Net이 프롬프트를 반영하는 방법 2번 · 어텐션 블록과 텐서 모양(실측 낱말 지도) |
| 7 | 풀링 벡터 vs 토큰 77줄 — 조건을 벡터 하나로 줄이면 | U-Net이 프롬프트를 반영하는 방법 2번 |
| 8 | 조건을 중간에 바꾸기 — 프롬프트가 언제 그림을 정하나 | U-Net과 타임스텝 4 · 5번 |
| 9 | CFG — 조건 있는 예측과 없는 예측의 차이를 키우기 · 부정 프롬프트 | CFG (1) ~ (3) |
| 10 | 조건 공간이 맞아야 — 같은 정보, 다른 좌표 | Diffusion 이미지 생성 1번 · U-Net이 프롬프트를 반영하는 방법 4번 |

- 실행 : **Run → Run All Cells**. GPU 필수. 전체 약 3~5분(RTX 4080). 셀 7 · 10 은 5-1 의 CLIP(약 600MB, 캐시에 있으면 받지 않음)으로 그림과 프롬프트의 일치도를 잰다
''')

md('''
### 셀 1 · 설정
''')
code('''
# 셀 1 · 설정
import os
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
SD_FILE  = f"{MODELS_DIR}/checkpoints/v1-5-pruned-emaonly.safetensors"
PROMPT   = "a red car on the beach"
PROMPT_B = "a blue boat on the sea"      # 셀 8 : 중간에 바꿔 넣을 프롬프트
SEED     = 42
STEPS    = 20
GUIDANCE = 7.5
WORDS    = ["red", "car", "beach"]        # 셀 6 : 어텐션 지도를 볼 낱말(PROMPT 안의 낱말)
POOL_PROMPTS = ["a red car on the beach", "a red cube on top of a blue sphere",
                "a cat wearing a yellow hat", "two dogs and one cat on a sofa"]   # 셀 7
NEGATIVE = "palm trees, clouds"           # 셀 9 : 부정 프롬프트
''')

md('''
### 셀 2 · 모델 불러오기 · 공통 함수
01 셀 2 와 같다. 덧붙여 `clip_score(그림, 글)` — 5-1 의 CLIP(ViT-B/32)으로 그림과 글의 코사인을 잰다(셀 7 · 10 에서 "프롬프트를 얼마나 따랐나"의 잣대).
''')
code('''
# 셀 2 · 모델 불러오기 · 공통 함수
import math, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic") if f in have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False})
torch.set_grad_enabled(False)
assert torch.cuda.is_available(), "GPU 가 보이지 않음 — 5-2 는 NVIDIA GPU 필수(README 「실행 환경」)"
DEV, DT = "cuda", torch.float16

pipe = StableDiffusionPipeline.from_single_file(SD_FILE, torch_dtype=DT).to(DEV)
pipe.set_progress_bar_config(disable=True)
tok, te, unet, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
SF = vae.config.scaling_factor
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"{torch.cuda.get_device_name(0)} · SD1.5 float16 · 잣대 CLIP ViT-B/32")


def encode(text, pooled=False):
    ids = tok([text], padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(DEV)
    out = te(ids)
    if pooled:                                   # [EOS] 한 줄(5-1 CLIP 이 검색에 쓴 벡터)을 77칸에 복제
        return out.pooler_output[:, None, :].expand(1, 77, -1).contiguous()
    return out.last_hidden_state


def noise(seed=SEED):
    return torch.randn((1, 4, 64, 64), generator=torch.Generator(DEV).manual_seed(seed), device=DEV, dtype=DT)


def to_img(z):
    x = vae.decode(z / SF).sample
    return Image.fromarray(((x[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))


def denoise(c, u, z=None, guidance=GUIDANCE, steps=STEPS, cond_at=None, each=None):
    """DDIM 디노이즈 루프. cond_at(i) 가 있으면 단계마다 조건을 바꿔 넣는다(셀 8)"""
    s = DDIMScheduler.from_config(pipe.scheduler.config); s.set_timesteps(steps)
    z = noise() if z is None else z
    for i, t in enumerate(s.timesteps):
        ci = cond_at(i) if cond_at else c
        if guidance == 1:                        # CFG 없음 : U-Net 한 번
            e = unet(z, t, encoder_hidden_states=ci).sample
        else:
            eu, ec = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([u, ci])).sample.chunk(2)
            e = eu + guidance * (ec - eu)
            if each: each(i, t, eu, ec)
        z = s.step(e, t, z).prev_sample
    return z


def clip_score(img, text):
    x = cproc(text=[text], images=img, return_tensors="pt", padding=True).to(DEV)
    o = clip(**x)
    return F.cosine_similarity(o.image_embeds, o.text_embeds).item()


def show(ims, titles=None, cols=None, size=2.4, suptitle=None):
    cols = cols or len(ims); rows = math.ceil(len(ims) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows + (0.3 if suptitle else 0)))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < len(ims):
            ax.imshow(ims[i])
            if titles: ax.set_title(titles[i], fontsize=8)
    if suptitle: fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout(); plt.show()


c, u = encode(PROMPT), encode("")
''')

md('''
### 셀 3 · U-Net 층별 모양
- 한 번 호출하면서 블록마다 들어가는 · 나오는 모양을 찍는다. 내려가는 길 64 → 32 → 16 → 8, 병목 8, 올라가는 길 8 → 16 → 32 → 64 — U 자
- 채널(칸 하나의 특징 수)은 반대로 320 → 640 → 1280 : 해상도를 줄인 만큼 칸마다 더 많은 특징을 담는다
- **skip** : 내려가는 길의 중간 결과 12개를 저장했다가 올라가는 길의 같은 해상도에서 **이어 붙인다**(채널 방향으로 쌓기). "skip 받은 뒤 채널" = 올라온 채널 + 보관본(skip) 채널(장표 「U-Net denoiser — skip과 학습」 3번)
- 1강의 잔차 연결은 **더하기**(모양 그대로), U-Net 의 skip 은 **이어붙이기**(채널이 늘어남) — 같은 우회로 발상, 다른 연산
''')
code('''
# 셀 3 · U-Net 층별 모양
rec, hs = [], []
def grab(name):
    def h(m, args, kwargs, o):
        x_in = kwargs.get("hidden_states", args[0] if args else None)
        x = o[0] if isinstance(o, tuple) else o
        n_skip = len(o[1]) if isinstance(o, tuple) and len(o) > 1 else 0
        attn = getattr(m, "attentions", None)
        rec.append((name, tuple(x_in.shape[1:]), tuple(x.shape[1:]), n_skip, bool(attn) and len(attn) > 0))
    return h
hs.append(unet.conv_in.register_forward_hook(grab("입력 conv"), with_kwargs=True))
for i, b in enumerate(unet.down_blocks): hs.append(b.register_forward_hook(grab(f"down {i}"), with_kwargs=True))
hs.append(unet.mid_block.register_forward_hook(grab("mid"), with_kwargs=True))
for i, b in enumerate(unet.up_blocks): hs.append(b.register_forward_hook(grab(f"up {i}"), with_kwargs=True))
hs.append(unet.conv_out.register_forward_hook(grab("출력 conv"), with_kwargs=True))
unet(noise(), torch.tensor(500, device=DEV), encoder_hidden_states=c)
for h in hs: h.remove()

fmt = lambda sh: f"{sh[0]} × {sh[1]}×{sh[2]}"
skip_in = {f"up {i}": [r.conv1.in_channels for r in b.resnets] for i, b in enumerate(unet.up_blocks)}
rows = [(n, fmt(a), fmt(b), "있음" if at else "–", k if n.startswith("down") else "", " · ".join(map(str, skip_in.get(n, []))))
        for n, a, b, k, at in rec]
print(pd.DataFrame(rows, columns=["블록", "들어감 (채널 × 해상도)", "나옴", "교차 어텐션", "skip 으로 저장", "skip 받은 뒤 채널"]).to_string(index=False))
print(f"\\nskip 으로 저장되는 텐서 : 입력 conv 1 + down 블록 {sum(r[3] for r in rec if r[0].startswith('down'))} = 12개 → 올라가는 길에서 하나씩 이어 붙임")
''')

md('''
### 셀 4 · skip 을 흐리게 하면
- 64×64 해상도로 넘어가는 skip 텐서를 **4×4 칸 평균으로 뭉갠 뒤 다시 늘려** 넘긴다 — 큰 구조는 남기고 칸 단위의 세부만 지움
  - ① 그대로 ② 마지막 5단계만 ③ 마지막 10단계만 ④ 20단계 전부
- 남은 노이즈 양 = 그림에서 저주파(16×16)를 뺀 나머지의 표준편차(01 셀 6 의 "디테일"과 같은 계산 — 깨끗한 그림은 작고, 노이즈가 남으면 커짐)
- 처음 예상 : 세부가 사라져 **흐린 그림**. 실제 : 흐려지지 않고 **노이즈가 지워지지 않고 남는다** → skip 이 나르는 세부 = "이 칸에 어떤 노이즈가 있는가". 깊은 층(8×8)만으로는 칸 단위 노이즈를 알 수 없어 지우지 못한다
''')
code('''
# 셀 4 · skip 을 흐리게 하면
ACTIVE = {"on": False}
def blur64(r, k=4):
    return F.interpolate(F.avg_pool2d(r, k), scale_factor=k, mode="bilinear") if r.shape[-1] == 64 else r
def pre(m, args, kwargs):
    if ACTIVE["on"]:
        kwargs["res_hidden_states_tuple"] = tuple(blur64(r) for r in kwargs["res_hidden_states_tuple"])
    return args, kwargs
hs = [b.register_forward_pre_hook(pre, with_kwargs=True) for b in unet.up_blocks]

def run_blur(active):
    s = DDIMScheduler.from_config(pipe.scheduler.config); s.set_timesteps(STEPS)
    z = noise()
    for i, t in enumerate(s.timesteps):
        ACTIVE["on"] = active(i)
        eu, ec = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([u, c])).sample.chunk(2)
        z = s.step(eu + GUIDANCE * (ec - eu), t, z).prev_sample
    ACTIVE["on"] = False
    return to_img(z)

def leftover(img):
    g = np.asarray(img.convert("L"), dtype=np.float32)
    low = np.asarray(Image.fromarray(g).resize((16, 16), Image.BOX).resize((512, 512), Image.BICUBIC), dtype=np.float32)
    return float((g - low).std())

ims, rows = [], []
for label, act in [("① 그대로", lambda i: False), ("② 마지막 5단계만", lambda i: i >= STEPS - 5),
                   ("③ 마지막 10단계만", lambda i: i >= STEPS - 10), ("④ 20단계 전부", lambda i: True)]:
    im = run_blur(act); ims.append(im)
    rows.append((label, f"{leftover(im):.1f}", f"{clip_score(im, PROMPT):.3f}"))
for h in hs: h.remove()
print(pd.DataFrame(rows, columns=["64×64 skip 을 흐린 구간", "남은 노이즈 양", "CLIP 일치도"]).to_string(index=False))
show(ims, [r[0] for r in rows], size=3.0)
''')

md('''
### 셀 5 · 교차 어텐션의 크기
- U-Net 안의 어텐션은 두 종류 : **셀프 어텐션**(그림 칸 ↔ 그림 칸, 1강과 같음) · **교차 어텐션**(그림 칸 → 글 토큰)
- 교차 어텐션 : **질의 Q 는 그림 칸**에서, **키 K · 값 V 는 글 토큰 77개**에서 만든다 → 확률표 = [그림 칸 수 × 77]. 그림의 칸마다 "77개 중 어느 낱말을 참고할까"를 softmax 로 정해 그 낱말들의 값을 섞어 가져온다
- 어텐션을 직접 계산하는 처리기로 바꿔 끼워 Q · K · 확률표 모양을 찍는다. 바꿔 끼워도 U-Net 출력은 같다(float16 반올림 차이만)
''')
code('''
# 셀 5 · 교차 어텐션의 크기
STORE = {"shapes": {}, "maps": None}

class Grab(AttnProcessor):
    """기본 어텐션과 같은 계산 + 모양 · 확률표 기록"""
    def __init__(self, name): self.name = name
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        cross = encoder_hidden_states is not None
        ctx = encoder_hidden_states if cross else hidden_states
        q, k, v = (attn.head_to_batch_dim(f(x)) for f, x in ((attn.to_q, hidden_states), (attn.to_k, ctx), (attn.to_v, ctx)))
        p = attn.get_attention_scores(q, k, None)
        STORE["shapes"].setdefault(self.name, (cross, attn.heads, tuple(q.shape), tuple(k.shape), tuple(p.shape)))
        if cross and STORE["maps"] is not None and q.shape[1] == 256:          # 16×16 해상도의 교차 어텐션만 모음(셀 6)
            STORE["maps"].append(p.view(-1, attn.heads, 256, 77)[-1].mean(0).float())   # 배치 마지막 = 조건 있는 쪽, 헤드 평균
        out = attn.batch_to_head_dim(torch.bmm(p, v))
        return attn.to_out[1](attn.to_out[0](out))

orig = unet.attn_processors
z, t = noise(), torch.tensor(500, device=DEV)
base = unet(z, t, encoder_hidden_states=c).sample
unet.set_attn_processor({n: Grab(n) for n in orig})
mine = unet(z, t, encoder_hidden_states=c).sample
print(f"직접 계산한 어텐션으로 바꿔도 U-Net 출력 최대 차이 {(base - mine).abs().max().item():.4f} (float16)\\n")

rows, seen = [], set()
for name, (cross, h, qs, ks, ps) in STORE["shapes"].items():
    key = (cross, qs[1])
    if key in seen: continue
    seen.add(key)
    side = int(qs[1] ** 0.5)
    rows.append(("교차" if cross else "셀프", f"{side}×{side}", h, f"{qs[1]} × {qs[2]}", f"{ks[1]} × {ks[2]}", f"{ps[1]} × {ps[2]}", f"{ps[1] * ps[2]:,}"))
T = pd.DataFrame(rows, columns=["종류", "해상도", "헤드", "Q (칸 × 헤드폭)", "K (토큰 × 헤드폭)", "확률표", "칸 수 / 헤드"])
print(T.sort_values(["종류", "해상도"]).to_string(index=False))
n_cross = sum(1 for v in STORE["shapes"].values() if v[0])
print(f"\\n교차 어텐션 층 {n_cross}개 · 셀프 어텐션 층 {len(STORE['shapes']) - n_cross}개 — 프롬프트는 이 {n_cross}곳에서 매 단계 들어간다")
''')

md('''
### 셀 6 · 단계마다 · 낱말마다 어텐션 지도
- 20단계 디노이즈 동안 16×16 해상도 교차 어텐션의 확률표를 모아 평균 → 낱말 하나의 열 = **그림의 어느 칸이 그 낱말을 참고했나**
- 지도는 낱말마다 최대값으로 나눠 밝기를 맞춘다. 표의 "확률 몫" = 그림 칸들이 그 낱말에 준 확률의 평균
- 첫 토큰(<|startoftext|>)이 확률의 대부분을 받는다 — 참고할 낱말이 없을 때 쉬어 가는 자리 역할. 낱말에 가는 몫은 작아도 위치가 뚜렷하다
''')
code('''
# 셀 6 · 낱말마다 어텐션 지도
STORE["maps"] = []
img = to_img(denoise(c, u))
A = torch.stack(STORE["maps"]).mean(0)                   # [256, 77]
STORE["maps"] = None
toks = tok.convert_ids_to_tokens(tok([PROMPT], padding="max_length", max_length=77).input_ids[0])
idx = {w: next(i for i, t in enumerate(toks) if t.replace("</w>", "") == w) for w in WORDS}
rows = [("<|startoftext|>", 0, f"{A[:, 0].mean():.3f}")] + [(w, i, f"{A[:, i].mean():.3f}") for w, i in idx.items()]
print(pd.DataFrame(rows, columns=["토큰", "자리", "확률 몫(그림 칸 평균)"]).to_string(index=False))
fig, axes = plt.subplots(1, len(WORDS) + 1, figsize=(3 * (len(WORDS) + 1), 3.2))
axes[0].imshow(img); axes[0].set_title(PROMPT, fontsize=8); axes[0].axis("off")
for ax, (w, i) in zip(axes[1:], idx.items()):
    m = A[:, i].view(16, 16).cpu().numpy(); m = m / m.max()
    ax.imshow(img); ax.imshow(np.asarray(Image.fromarray((m * 255).astype("uint8")).resize((512, 512), Image.BICUBIC)), cmap="jet", alpha=0.5)
    ax.set_title(f"'{w}' 를 참고한 칸", fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
unet.set_attn_processor(orig)
''')

md('''
### 셀 7 · 풀링 벡터 vs 토큰 77줄
- 같은 프롬프트를 두 방식으로 넣는다 : ① **토큰 77줄**(SD1.5 가 학습한 방식) ② **[EOS] 풀링 벡터 하나를 77칸에 복제** — 5-1 CLIP 이 검색에 쓴 바로 그 벡터
- ② 에서는 그림의 모든 칸이 같은 벡터만 보게 된다 → 낱말별로 골라 볼 수 없음
- 잣대 : CLIP 일치도(그림 ↔ 프롬프트 코사인, 5-1 01 셀 6 과 같은 계산). 값 자체보다 ①과 ②의 차이를 본다
- 주의 : U-Net 은 ① 방식으로만 학습했다 → ②의 저하에는 "정보가 뭉개짐"과 "학습 때 못 본 입력"이 겹친다
''')
code('''
# 셀 7 · 풀링 벡터 vs 토큰 77줄
ims, titles, rows = [], [], []
for p in POOL_PROMPTS:
    a = to_img(denoise(encode(p), u))
    b = to_img(denoise(encode(p, pooled=True), encode("", pooled=True)))
    sa, sb = clip_score(a, p), clip_score(b, p)
    rows.append((p, f"{sa:.3f}", f"{sb:.3f}"))
    ims += [a, b]; titles += [f"토큰 77줄 · {sa:.3f}", f"풀링 1개 · {sb:.3f}"]
print(pd.DataFrame(rows, columns=["프롬프트", "① 토큰 77줄", "② 풀링 1개 복제"]).to_string(index=False))
show(ims, titles, cols=4, size=2.6)
''')

md('''
### 셀 8 · 조건을 중간에 바꾸기
- 같은 시작 노이즈로 `PROMPT` 로 출발해 **k 단계 뒤부터** `PROMPT_B` 로 바꿔 넣는다(20단계 중)
- k 가 작으면 PROMPT_B 의 그림, k 가 크면 PROMPT 의 구도에 PROMPT_B 의 색 · 질감이 덧칠 — 어느 k 에서 구도가 넘어가지 않게 되나
- 장표 「U-Net과 타임스텝」 4 · 5번 "노이즈가 큰 단계는 조건 c 가 큰 구조를, 작은 단계는 현재 z_t 가 세부를" + 01 셀 6 의 구도 일치 곡선과 같은 이야기
''')
code('''
# 셀 8 · 조건을 중간에 바꾸기
cB = encode(PROMPT_B)
K = [0, 1, 2, 3, 5, 8, 12, 20]
ims = [to_img(denoise(c, u, cond_at=lambda i, k=k: c if i < k else cB)) for k in K]
show(ims, [f"k = {k}" + (" (처음부터 B)" if k == 0 else " (끝까지 A)" if k == STEPS else "") for k in K], cols=len(K), size=2.0,
     suptitle=f"A = {PROMPT} → k 단계 뒤 B = {PROMPT_B}")
''')

md('''
### 셀 9 · CFG — 조건 있는 예측과 없는 예측의 차이를 키우기
- 학습 : SD1.5 는 학습 중 **10% 확률로 프롬프트를 비우고** 같은 손실을 학습했다(모델카드) → 한 U-Net 이 "조건 있는 예측 ε_c" 와 "조건 없는 예측 ε_u" 를 둘 다 낼 줄 안다
- 생성 : 매 단계 두 번 예측해 `ε = ε_u + w·(ε_c − ε_u)` — 둘의 차이(= 프롬프트가 미는 방향)를 w 배로. w = 1 이면 조건 있는 예측 그대로
- ① 두 예측의 차이는 얼마나 작나(단계별 `|ε_c − ε_u| / |ε_u|`) ② w 별 결과 · 시간 ③ **부정 프롬프트** = 빈 프롬프트 자리에 넣은 글 → "그쪽에서 멀어지는" 방향으로 증폭
''')
code('''
# 셀 9 · CFG
ratio = []
denoise(c, u, each=lambda i, t, eu, ec: ratio.append((i + 1, int(t), ((ec - eu).float().norm() / eu.float().norm()).item())))
R = pd.DataFrame(ratio, columns=["단계", "t", "|ε_c − ε_u| / |ε_u|"])
print(R.iloc[[0, 4, 9, 14, 19]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

W = [1, 3, 7.5, 15]
ims, titles = [], []
for w in W:
    torch.cuda.synchronize(); t0 = time.time()
    im = to_img(denoise(c, u, guidance=w)); torch.cuda.synchronize()
    ims.append(im); titles.append(f"w = {w} · {time.time() - t0:.2f}초 · CLIP {clip_score(im, PROMPT):.3f}")
neg = to_img(denoise(c, encode(NEGATIVE)))
ims.append(neg); titles.append(f"w = {GUIDANCE} · 부정 '{NEGATIVE}'")
show(ims, titles, cols=len(ims), size=2.6)
''')

md('''
### 셀 10 · 조건 공간이 맞아야
- U-Net 은 글 인코더가 만든 768개 값을 **그 순서(좌표) 그대로** 읽도록 학습됐다
- ① 그대로 ② 같은 768개 값을 **칸 순서만 섞어** 넣기(정보량은 같음) ③ 다른 글 인코더 — 5-1 의 CLIP ViT-B/32 텍스트 타워(폭 512)
- 장표 「Diffusion 이미지 생성」 1번 "인코더만 갈아끼우면 안 됨, U-Net 과 조건 공간이 맞아야" · 「U-Net이 프롬프트를 반영하는 방법」 4번
''')
code('''
# 셀 10 · 조건 공간이 맞아야
perm = torch.randperm(768, generator=torch.Generator().manual_seed(0)).to(DEV)
a = to_img(denoise(c, u))
b = to_img(denoise(c[..., perm], u[..., perm]))
rows = [("① 그대로", f"{clip_score(a, PROMPT):.3f}"), ("② 768칸 순서만 섞음", f"{clip_score(b, PROMPT):.3f}")]
small_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
small = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=DT).to(DEV)
cs = small(small_tok([PROMPT], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(DEV)).last_hidden_state
try:
    unet(noise(), torch.tensor(500, device=DEV), encoder_hidden_states=cs)
    msg = "실행됨"
except Exception as e:
    msg = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
rows.append((f"③ 5-1 CLIP 텍스트 타워 {tuple(cs.shape)}", msg))
print(pd.DataFrame(rows, columns=["조건", "CLIP 일치도 / 결과"]).to_string(index=False))
show([a, b], ["① 그대로", "② 같은 값, 칸 순서만 섞음"], size=3.2)
''')

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT, len(cells), 'cells')
