"""5-2 실습 노트북 01 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook_01.py   → ../notebooks/01_디퓨전_원리.ipynb (출력 없이)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / '01_디퓨전_원리.ipynb'
cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': s.strip('\n').splitlines(keepends=True)})


md('''
# 실습 01 · 디퓨전 원리 — SD1.5 를 부품으로 나눠 직접 돌리기

장표 「Diffusion 개념」~「sampler」를 Stable Diffusion 1.5 로 확인한다. 파이프라인 한 줄 대신 **부품(글 인코더 · U-Net · VAE · 스케줄러)을 하나씩 꺼내** 텐서 모양 · 정방향 노이즈 · 학습 손실 · 디노이즈 루프를 손으로 따라간다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | 모델 불러오기 · 부품별 파라미터 수 | Diffusion 디노이저 1번 · 파이프라인 도해 |
| 3 | 파이프라인 텐서 흐름 — 프롬프트 · 시드 → U-Net → VAE 디코더 | 파이프라인 도해 |
| 4 | 정방향 — 사진에 노이즈를 단계별로 섞기 | Diffusion 개념 1 · 4번 · scheduler/sampler의 역할 3번 · U-Net과 타임스텝 1번 |
| 5 | 학습 한 걸음 — 섞은 노이즈를 맞히는 손실, t 별 크기 | scheduler/sampler의 역할 · U-Net denoiser 5번 |
| 6 | 단계마다 U-Net 이 추정한 완성본 x̂₀ — 구도와 디테일이 정해지는 순서 | U-Net과 타임스텝 4~7번 |
| 7 | 손으로 쓴 디노이즈 루프 = 파이프라인 결과 · 이동 규칙 몇 줄 | scheduler/sampler의 역할 · 샘플링 — 디노이즈 1회의 계산 |
| 8 | 샘플러 비교 — DDPM · DDIM · Euler · Euler a · DPM++ × 스텝 수 | sampler |
| 9 | timestep 임베딩 — 사인 · 코사인 벡터 | U-Net과 타임스텝 6번 |

- 실행 : **Run → Run All Cells**. GPU 필수(VRAM 4GB 이상). 전체 약 3~5분(RTX 4080)
- 모델 : SD1.5 체크포인트 한 파일(`v1-5-pruned-emaonly.safetensors`, 4.3GB) — ComfyUI 와 같은 파일을 읽기 전용으로 씀. 없으면 `tools/download_models.sh`
- 모든 그림은 같은 시드(셀 1 `SEED`)에서 출발 → 다시 돌려도 같은 그림
''')

md('''
### 셀 1 · 설정
- `SD_FILE` : SD1.5 체크포인트 경로(도커 기본 `/models/checkpoints/...`)
- `PROMPT` · `SEED` · `STEPS` · `GUIDANCE` : 셀 3 · 6 · 7 · 8 의 생성 조건. `GUIDANCE`(조건을 따르는 세기)는 02 셀 9 에서 다룬다 — 여기서는 파이프라인 기본값 7.5
- `PHOTO` : 셀 4 에서 노이즈를 섞을 사진(5-1 사진 150장 중 하나)
''')
code('''
# 셀 1 · 설정
import os
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
SD_FILE  = f"{MODELS_DIR}/checkpoints/v1-5-pruned-emaonly.safetensors"
PROMPT   = "a red car on the beach"
SEED     = 42
STEPS    = 20
GUIDANCE = 7.5
PHOTO    = "pet_samoyed_1.jpg"      # 5-1/images 안의 파일 이름
''')

md('''
### 셀 2 · 모델 불러오기 · 부품별 파라미터 수
- 체크포인트 한 파일 안에 부품 넷이 들어 있다 : **글 인코더**(CLIP 텍스트 타워 — 5-1 의 그것) · **U-Net**(디노이저) · **VAE**(압축기) · **스케줄러**(웨이트 없는 이동 규칙)
- 파라미터 대부분이 U-Net — "핵심은 반복 디노이즈, 나머지는 장치"(장표 「Diffusion 디노이저」 1번)를 숫자로
- 공통 함수 : `encode`(프롬프트 → 조건 행렬) · `to_img`(latent → 그림) · `enc_img`(그림 → latent) · `denoise`(디노이즈 루프)
''')
code('''
# 셀 2 · 모델 불러오기 · 부품별 파라미터 수
import math, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from diffusers import (StableDiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler,
                       EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler)

have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic") if f in have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False})
torch.set_grad_enabled(False)
assert torch.cuda.is_available(), "GPU 가 보이지 않음 — 5-2 는 NVIDIA GPU 필수(README 「실행 환경」)"
DEV, DT = "cuda", torch.float16
IMAGE_DIR = next(p for p in (Path("/work/5-1/images"), Path.cwd().parents[1] / "5-1" / "images") if p.exists())

t0 = time.time()
pipe = StableDiffusionPipeline.from_single_file(SD_FILE, torch_dtype=DT).to(DEV)
pipe.set_progress_bar_config(disable=True)
tok, te, unet, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
SF = vae.config.scaling_factor
print(f"불러오기 {time.time() - t0:.1f}초 · {torch.cuda.get_device_name(0)} · float16")

parts = [("글 인코더 (CLIP 텍스트 타워)", te), ("U-Net (디노이저)", unet), ("VAE (압축기)", vae)]
n = {k: sum(p.numel() for p in m.parameters()) for k, m in parts}
tot = sum(n.values())
print(pd.DataFrame([(k, f"{v / 1e6:,.1f}M", f"{v / tot:.1%}") for k, v in n.items()] + [("합계", f"{tot / 1e6:,.1f}M", "100%")],
                   columns=["부품", "파라미터", "비율"]).to_string(index=False))
print(f"스케줄러 : {type(pipe.scheduler).__name__} — 웨이트 0개. 학습 때 노이즈 단계 {pipe.scheduler.config.num_train_timesteps}개")


def encode(text):
    """프롬프트 → 조건 행렬 c [1, 77, 768] (5-1 텍스트 타워에서 풀링하기 전 77줄)"""
    ids = tok([text], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
    return te(ids.to(DEV))[0]


def to_img(z):
    """latent [1, 4, 64, 64] → 그림 (VAE 디코더)"""
    x = vae.decode(z / SF).sample
    return Image.fromarray(((x[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))


def enc_img(im):
    """그림 → latent (VAE 인코더의 평균값 × 스케일 계수)"""
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(DEV, DT)
    return vae.encode(x).latent_dist.mean * SF


def square(path, size=512):
    im = Image.open(path).convert("RGB"); s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((size, size), Image.BICUBIC)


def noise(seed=SEED):
    return torch.randn((1, 4, 64, 64), generator=torch.Generator(DEV).manual_seed(seed), device=DEV, dtype=DT)


def denoise(sched, c, u, z, guidance=GUIDANCE, steps=STEPS, seed=SEED, each=None):
    """디노이즈 루프. sched = 이동 규칙, U-Net 은 매 단계 노이즈 방향만 예측"""
    sched.set_timesteps(steps)
    z = z * sched.init_noise_sigma
    g = torch.Generator(DEV).manual_seed(seed + 10_000)  # 매 단계 다시 섞는 노이즈(DDPM · Euler a)용. 시작 노이즈와 다른 수열이어야 함
    for i, t in enumerate(sched.timesteps):
        zin = sched.scale_model_input(torch.cat([z, z]), t)
        eu, ec = unet(zin, t, encoder_hidden_states=torch.cat([u, c])).sample.chunk(2)
        e = eu + guidance * (ec - eu)                    # 조건 없는 예측 ↔ 조건 있는 예측 차이를 증폭 (02 셀 9)
        out = sched.step(e, t, z, generator=g)
        if each: each(i, t, z, e, out)
        z = out.prev_sample
    return z


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
''')

md('''
### 셀 3 · 파이프라인 텐서 흐름
장표 「파이프라인 도해」의 화살표마다 실제 텐서 모양을 찍는다.
- ① 프롬프트 → 토큰 번호 77칸 → ② 글 인코더 → 조건 행렬 c **[1, 77, 768]** — 5-1 CLIP(B/32)은 [77 × 512] 에서 [EOS] 한 줄만 꺼냈다. 여기서는 **77줄을 전부** U-Net 에 넘긴다(02 셀 5 · 7)
- ③ 시드 → 시작 노이즈 z_T **[1, 4, 64, 64]** — 그림(512×512×3)이 아니라 VAE 의 latent 모양
- ④ U-Net(z_t, t, c) → 예측 노이즈 ε̂ — **입력과 같은 모양**. U-Net 은 "지울 노이즈"만 내놓는다
- ⑤ 이 호출을 `STEPS` 번 반복(셀 7) → z_0 → ⑥ VAE 디코더 → 그림 **[1, 3, 512, 512]**
''')
code('''
# 셀 3 · 파이프라인 텐서 흐름
ids = tok([PROMPT], padding="max_length", max_length=77, return_tensors="pt").input_ids
c, u = encode(PROMPT), encode("")
zT = noise()
t = torch.tensor(999, device=DEV)
eps = unet(zT, t, encoder_hidden_states=c).sample
sched = DDIMScheduler.from_config(pipe.scheduler.config)
t0 = time.time(); z0 = denoise(sched, c, u, noise()); torch.cuda.synchronize(); dt = time.time() - t0
x = vae.decode(z0 / SF).sample

words = tok.convert_ids_to_tokens(ids[0, :8].tolist())
rows = [("① 토큰 번호", tuple(ids.shape), f"실제 토큰 {int((ids != tok.pad_token_id).sum()) + 1}개 + 나머지 [EOS]·패딩 : {' '.join(words[:7])} …"),
        ("② 조건 c = 글 인코더 출력", tuple(c.shape), "77줄 전부 — 풀링하지 않음"),
        ("③ 시작 노이즈 z_T (시드)", tuple(zT.shape), f"값 {zT.numel():,}개 · 평균 {zT.float().mean():+.3f} · 표준편차 {zT.float().std():.3f}"),
        ("④ U-Net(z_T, t=999, c) → ε̂", tuple(eps.shape), "입력과 같은 모양 = 지울 노이즈"),
        (f"⑤ {STEPS}단계 반복 → z_0", tuple(z0.shape), f"U-Net 호출 {STEPS}×2회(조건 있음·없음) · {dt:.2f}초"),
        ("⑥ VAE 디코더 → 그림", tuple(x.shape), f"값 {x.numel():,}개 = latent 의 {x.numel() / z0.numel():.0f}배")]
print(pd.DataFrame(rows, columns=["단계", "모양", "메모"]).to_string(index=False))
show([to_img(zT), to_img(z0)], ["③ z_T 를 그대로 디코드(노이즈)", f"⑥ 결과 : {PROMPT}"], size=3.4)
''')

md('''
### 셀 4 · 정방향 — 사진에 노이즈를 단계별로 섞기
- 학습 데이터 쪽에서 본 디퓨전 : 깨끗한 latent z₀ 에 정해진 비율로 노이즈 ε 를 섞는다
  `z_t = a_t · z₀ + b_t · ε` — a_t² + b_t² = 1, t 가 커질수록 a 는 줄고 b 는 는다(**노이즈 스케줄**)
- 이 식에는 학습할 것이 없다 — 표의 a · b 는 숫자 두 줄(스케줄러 설정)에서 계산된다(장표 「Diffusion 개념」 4번 "정방향은 고정 규칙")
- 섞은 ε 를 우리가 알고 있으므로 (z_t, ε) 쌍 = **정답이 붙은 학습 문제**가 공짜로 무한히 나온다
''')
code('''
# 셀 4 · 정방향 — 사진에 노이즈를 단계별로 섞기
ddpm = DDPMScheduler.from_config(pipe.scheduler.config)
abar = ddpm.alphas_cumprod.to(DEV)
photo = square(IMAGE_DIR / PHOTO)
z0p = enc_img(photo)
eps0 = noise(SEED + 1)
T = [0, 50, 150, 300, 500, 700, 850, 999]
ims, rows = [], []
for t in T:
    a, b = abar[t].sqrt(), (1 - abar[t]).sqrt()
    zt = a * z0p + b * eps0
    ims.append(to_img(zt.to(DT)))
    rows.append((t, f"{a:.3f}", f"{b:.3f}", f"{(a / b) ** 2:,.2f}"))
print(pd.DataFrame(rows, columns=["t", "a_t (원본 비율)", "b_t (노이즈 비율)", "신호/노이즈 a²/b²"]).to_string(index=False))
print(f"스케줄 : beta {ddpm.config.beta_start} → {ddpm.config.beta_end} ({ddpm.config.beta_schedule}), 단계 {ddpm.config.num_train_timesteps}개")
show(ims, [f"t = {t}" for t in T], size=2.0, suptitle=f"{PHOTO} 의 latent 에 노이즈를 섞어 디코드")
''')

md('''
### 셀 5 · 학습 한 걸음 — 섞은 노이즈를 맞히는 손실
- 학습 한 걸음 = 사진 하나 · 무작위 t · 무작위 ε → z_t 를 만들고 → U-Net 이 ε̂ 를 예측 → **손실 = (ε − ε̂)² 의 평균**. 정답 ε 는 셀 4 처럼 우리가 섞은 것
- 이미 학습된 SD1.5 로 이 손실을 t 별로 재 본다(사진 8장 × t 마다 노이즈 4번)
  - **ε 손실** : 학습 목표 그대로
  - **x₀ 손실** : 같은 예측을 "원본 추정" `x̂₀ = (z_t − b·ε̂) / a` 로 바꿔 원본과 비교
- 읽는 법 : 같은 목표(ε 맞히기)인데 t 마다 손실 크기가 다르다 = **t 마다 다른 문제**(장표 「U-Net과 타임스텝」 3번)
''')
code('''
# 셀 5 · 학습 한 걸음 — 섞은 노이즈를 맞히는 손실
meta = pd.read_csv(IMAGE_DIR / "images.csv")
pets = meta[meta.group == "pet"].groupby("label").head(1).head(8)
batch = [(enc_img(square(IMAGE_DIR / f)), encode(f"a photo of a {lab.replace('_', ' ')}")) for f, lab in zip(pets.file, pets.label)]
T5 = [20, 100, 250, 400, 550, 700, 850, 980]
g = torch.Generator(DEV).manual_seed(SEED)
rows = []
for t in T5:
    a, b = abar[t].sqrt(), (1 - abar[t]).sqrt()
    le, lx = [], []
    for z0b, cb in batch:
        for _ in range(4):
            e = torch.randn(z0b.shape, generator=g, device=DEV, dtype=DT)
            zt = (a * z0b + b * e).to(DT)
            eh = unet(zt, torch.tensor(t, device=DEV), encoder_hidden_states=cb).sample
            le.append(((eh.float() - e.float()) ** 2).mean().item())
            lx.append((((zt.float() - b * eh.float()) / a - z0b.float()) ** 2).mean().item())
    rows.append((t, np.mean(le), np.mean(lx)))
L = pd.DataFrame(rows, columns=["t", "ε 손실", "x₀ 손실"])
print(L.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(L.t, L["ε 손실"], "o-"); ax[0].set_title("ε 손실 (학습 목표)", fontsize=9); ax[0].set_xlabel("t")
ax[1].plot(L.t, L["x₀ 손실"], "o-", color="C1"); ax[1].set_title("같은 예측을 원본 추정으로 바꾼 손실", fontsize=9); ax[1].set_xlabel("t")
plt.tight_layout(); plt.show()
''')

md('''
### 셀 6 · 단계마다 U-Net 이 추정한 완성본 x̂₀
- 디노이즈 중간에 "지금 예측대로라면 완성본은 이것" `x̂₀` 를 디코드한다(스케줄러가 매 단계 계산하는 값)
- 윗줄 = 현재 상태 z_t(노이즈 섞인 그대로), 아랫줄 = 그 단계의 완성본 추정
- 표 : 각 단계 추정이 최종 그림과 얼마나 같은가를 **구도(저주파 — 16×16 으로 줄인 그림)** 와 **디테일(고주파 — 원본에서 저주파를 뺀 나머지)** 로 나눠 상관계수로 잰다
- 장표 「U-Net과 타임스텝」 7번 "초반 윤곽 · 후반 디테일은 명령한 규칙이 아님" — 아무도 시키지 않았는데 구도가 먼저 정해지는지 숫자로
''')
code('''
# 셀 6 · 단계마다 U-Net 이 추정한 완성본 x̂₀
snap = {}
def keep(i, t, z, e, out):
    snap[i] = (int(t), z.clone(), out.pred_original_sample.clone())
zf = denoise(DDIMScheduler.from_config(pipe.scheduler.config), c, u, noise(), each=keep)
final = np.asarray(to_img(zf).convert("L"), dtype=np.float32)


def bands(img):
    g = np.asarray(img.convert("L"), dtype=np.float32)
    low = np.asarray(Image.fromarray(g).resize((16, 16), Image.BOX).resize((512, 512), Image.BICUBIC), dtype=np.float32)
    return low, g - low


fl, fh = bands(to_img(zf))
pick = [0, 2, 4, 7, 10, 14, STEPS - 1]
top, bot, rows = [], [], []
for i in range(STEPS):
    t, zt, x0 = snap[i]
    im = to_img(x0)
    l, h = bands(im)
    rows.append((i + 1, t, np.corrcoef(l.ravel(), fl.ravel())[0, 1], np.corrcoef(h.ravel(), fh.ravel())[0, 1]))
    if i in pick:
        top.append(to_img(zt)); bot.append(im)
R = pd.DataFrame(rows, columns=["단계", "t", "구도 일치", "디테일 일치"])
print(R.iloc[pick].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
show(top + bot, [f"{i + 1}단계 z_t" for i in pick] + [f"{i + 1}단계 완성본 추정" for i in pick], cols=len(pick), size=1.9)
fig, ax = plt.subplots(figsize=(6, 2.6))
ax.plot(R["단계"], R["구도 일치"], "o-", label="구도(저주파)"); ax.plot(R["단계"], R["디테일 일치"], "s-", label="디테일(고주파)")
ax.set_xlabel("단계"); ax.set_ylabel("최종 그림과 상관"); ax.legend(fontsize=8); plt.tight_layout(); plt.show()
''')

md('''
### 셀 7 · 손으로 쓴 디노이즈 루프 = 파이프라인
- 셀 2 의 `denoise` (20줄 남짓) 결과와 diffusers 파이프라인 한 줄 결과를 같은 시작 노이즈로 비교 → **최대 차이 0** 이면 파이프라인이 하는 일이 이 루프 그대로
- U-Net 은 매 단계 ε̂ 만 낸다. 다음 상태로 **실제로 옮기는 것**은 스케줄러의 `step` — DDIM 의 step 을 직접 쓴 두 줄과도 비교한다
  `x̂₀ = (z_t − b_t·ε̂) / a_t` → `z_다음 = a_다음·x̂₀ + b_다음·ε̂`
- 장표 「scheduler/sampler의 역할」 1 · 2번 : U-Net 은 방향, 스케줄러는 이동 규칙
- 장표 「샘플링 — 디노이즈 1회의 계산」 : 밟을 t 목록(951 → 901 → … → 1)과 두 줄 식을 아래 출력으로 확인
''')
code('''
# 셀 7 · 손으로 쓴 디노이즈 루프 = 파이프라인
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
zp = pipe(PROMPT, num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=noise(), output_type="latent").images
zm = denoise(DDIMScheduler.from_config(pipe.scheduler.config), c, u, noise())
print(f"파이프라인 vs 손으로 쓴 루프 : 최대 차이 {(zp - zm).abs().max().item():.6f}")

s = DDIMScheduler.from_config(pipe.scheduler.config); s.set_timesteps(STEPS)
z = noise(); t, tn = s.timesteps[0], s.timesteps[1]
eu, ec = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([u, c])).sample.chunk(2)
e = eu + GUIDANCE * (ec - eu)
ref = s.step(e, t, z).prev_sample
a, b = s.alphas_cumprod[t].sqrt(), (1 - s.alphas_cumprod[t]).sqrt()
an, bn = s.alphas_cumprod[tn].sqrt(), (1 - s.alphas_cumprod[tn]).sqrt()
x0h = (z - b * e) / a                     # 라이브러리와 같은 float16 으로 계산
mine = an * x0h + bn * e
print(f"DDIM step(라이브러리) vs 두 줄 : t {int(t)} → {int(tn)}, 최대 차이 {(ref - mine).abs().max().item():.6f}")
print(f"샘플러가 밟는 t ({STEPS}개) : {s.timesteps.tolist()}")
''')

md('''
### 셀 8 · 샘플러 비교
- 같은 U-Net · 같은 프롬프트 · 같은 시작 노이즈, **이동 규칙(샘플러)만** 바꾼다 × 스텝 10 · 20 · 50
  - DDPM : 학습 때의 확률적 역과정, 매 단계 노이즈를 다시 섞음
  - DDIM : 노이즈를 다시 섞지 않는 결정론적 이동
  - Euler : 미분방정식을 한 걸음씩 적분(1차)
  - Euler a(ancestral) : Euler + 매 단계 노이즈 재주입
  - DPM++ 2M : 앞 단계 기울기까지 쓰는 고차 적분
- 표 ① 시간 ② **같은 시드로 다시 돌리면 같은가**(재현) ③ **10스텝과 50스텝의 구도가 같은가**(셀 6 과 같은 저주파 상관) ④ **50스텝 결과가 DDIM 50스텝과 같은 그림인가**
- 읽는 법 : 노이즈를 다시 섞지 않는 샘플러(DDIM · Euler · DPM++)는 같은 시작 노이즈에서 **같은 그림으로 모인다**(④). 다시 섞는 샘플러(DDPM · Euler a)는 다른 그림 — 같은 시드면 재현은 된다(②)
- 장표 「sampler」 2번 DDIM 의 "같은 seed 면 같은 결과"는 DDIM 만의 장점이 아님(②)
''')
code('''
# 셀 8 · 샘플러 비교
SAMPLERS = {"DDPM": DDPMScheduler, "DDIM": DDIMScheduler, "Euler": EulerDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler, "DPM++ 2M": DPMSolverMultistepScheduler}
SN = [10, 20, 50]
ims, titles, rows, fin = [], [], [], {}
for name, cls in SAMPLERS.items():
    res = {}
    for n in SN:
        torch.cuda.synchronize(); t0 = time.time()
        z = denoise(cls.from_config(pipe.scheduler.config), c, u, noise(), steps=n)
        torch.cuda.synchronize(); res[n] = (z, time.time() - t0)
    again = denoise(cls.from_config(pipe.scheduler.config), c, u, noise(), steps=10)
    l10, _ = bands(to_img(res[10][0])); l50, _ = bands(to_img(res[50][0]))
    rows.append([name, f"{res[10][1]:.2f}", f"{res[50][1]:.2f}", (again - res[10][0]).abs().max().item() == 0,
                 f"{np.corrcoef(l10.ravel(), l50.ravel())[0, 1]:.3f}"])
    fin[name] = bands(to_img(res[50][0]))[0]
    for n in SN:
        ims.append(to_img(res[n][0])); titles.append(f"{name} · {n}스텝")
ref = fin["DDIM"]
for r in rows:
    r.append(f"{np.corrcoef(fin[r[0]].ravel(), ref.ravel())[0, 1]:.3f}")
print(pd.DataFrame(rows, columns=["샘플러", "10스텝 초", "50스텝 초", "같은 시드 재현", "구도 일치(10↔50)", "50스텝 구도 vs DDIM"]).to_string(index=False))
show(ims, titles, cols=len(SN), size=2.2)
''')

md('''
### 셀 9 · timestep 임베딩 — 사인 · 코사인 벡터
- U-Net 은 숫자 t 를 그대로 받지 않는다. t → **사인 · 코사인 320개** → 작은 신경망 → **1280차원 벡터** → 모든 블록에 더해짐(장표 「U-Net과 타임스텝」 6번)
- 사인 · 코사인 부분은 1강 트랜스포머의 위치 인코딩과 같은 식 — 위치 번호 대신 t 를 넣었다. 직접 계산한 값과 비교해 최대 차이 0
- 그림 : 가로 = 320칸, 세로 = t(0~999). 앞쪽 칸은 천천히, 뒤쪽 칸은 빨리 변해 t 마다 다른 무늬
''')
code('''
# 셀 9 · timestep 임베딩 — 사인 · 코사인 벡터
ts = torch.arange(0, 1000, device=DEV)
tp = unet.time_proj(ts)                                   # 사인 · 코사인 [1000, 320]
temb = unet.time_embedding(tp.to(DT))                     # 작은 신경망 → [1000, 1280]
half = tp.shape[1] // 2
freq = torch.exp(-math.log(10000) * torch.arange(half, device=DEV) / (half - unet.config.freq_shift))
mine = torch.cat([torch.cos(ts[:, None] * freq), torch.sin(ts[:, None] * freq)], dim=1)   # flip_sin_to_cos=True → cos 먼저
print(f"time_proj {tuple(tp.shape)} → time_embedding {tuple(temb.shape)}")
print(f"직접 계산한 사인 · 코사인과 최대 차이 : {(tp - mine).abs().max().item():.2e}  (flip_sin_to_cos={unet.config.flip_sin_to_cos}, freq_shift={unet.config.freq_shift})")
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].imshow(tp.cpu().numpy(), aspect="auto", cmap="RdBu"); ax[0].set_title("사인·코사인 [t × 320]", fontsize=9); ax[0].set_ylabel("t"); ax[0].set_xlabel("칸")
sim = torch.nn.functional.cosine_similarity(temb[::50, None].float(), temb[None, ::50].float(), dim=-1).cpu().numpy()
ax[1].imshow(sim, cmap="viridis"); ax[1].set_title("1280차원 벡터끼리 코사인 (t = 0, 50, …, 950)", fontsize=9)
ax[1].set_xticks(range(0, 20, 4)); ax[1].set_xticklabels([str(v) for v in range(0, 1000, 200)])
ax[1].set_yticks(range(0, 20, 4)); ax[1].set_yticklabels([str(v) for v in range(0, 1000, 200)])
plt.tight_layout(); plt.show()
''')

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT, len(cells), 'cells')
