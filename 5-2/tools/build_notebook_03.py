"""5-2 실습 노트북 03 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook_03.py   → ../notebooks/03_VAE와_latent.ipynb (출력 없이)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / '03_VAE와_latent.ipynb'
cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': s.strip('\n').splitlines(keepends=True)})


md('''
# 실습 03 · VAE와 latent — 압축 장치를 열어 보기

장표 「VAE」 · 「Latent 채널」을 확인한다(셀 5 · 6 의 스케일 계수 · KL 은 강의 장표에 없는 보충 실습). 셀 3~7 은 SD1.5 의 VAE 를 Jupyter 에서 직접, 셀 8 은 **ComfyUI 를 API 로 부려** 4채널 · 16채널 VAE 네 개를 같은 문서로 비교한다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | VAE 불러오기 · ComfyUI 연결 | – |
| 3 | 압축 — 그림 [3, 512, 512] → latent [4, 64, 64], 48배 · 되살린 그림 | VAE 2 · 3번 · Latent 채널 2번 |
| 4 | latent 4채널을 채널별로 보기 — RGB 가 아님 | Latent 채널 1번 |
| 5 | latent 의 크기 — 표준편차와 스케일 계수 0.18215 | 보충(강의 장표 없음) — 디퓨전이 섞는 노이즈와 latent 크기 |
| 6 | 인코더가 내는 μ · σ — 분포로 만든다는 것, KL 이 얼마나 약한가 · reparameterization | 보충(강의 장표 없음) |
| 7 | latent 한 칸을 바꾸면 그림의 어디가 바뀌나 | Latent 채널 2번 |
| 8 | 글자 복원 — SD1.5 · SDXL(4채널) vs FLUX · Qwen-Image(16채널) VAE 를 ComfyUI 로 | VAE 4번 · Latent 채널(16채널로 간 이유) |

- 실행 : **Run → Run All Cells**. GPU 필수. 셀 8 은 ComfyUI(`docker compose` 의 comfy 서비스)가 떠 있어야 한다. 전체 약 2~3분
''')

md('''
### 셀 1 · 설정
- `PHOTOS` : 셀 3 · 5 · 6 에 쓸 사진(5-1 사진 150장 중 반려동물 40장)
- `DOC` · `DOC_WIDTH` : 셀 8 의 문서. 폭을 줄여 글자를 작게 만든다(512 → 글자 높이 약 11픽셀)
- `VAES` : 셀 8 에서 비교할 VAE (ComfyUI 의 models/vae · checkpoints 파일 이름)
''')
code('''
# 셀 1 · 설정
import os
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
COMFY_URL  = os.environ.get("COMFY_URL", "http://comfy:8188")
SD_FILE    = f"{MODELS_DIR}/checkpoints/v1-5-pruned-emaonly.safetensors"
PHOTO      = "pet_samoyed_1.jpg"
DOC        = "doc_3_pump_inspection.png"     # 5-1/docs 의 문서
DOC_WIDTH  = 512
VAES = {"SD1.5 (4채널)":        ("ckpt", "v1-5-pruned-emaonly.safetensors"),
        "SDXL (4채널)":         ("vae",  "sdxl_vae.safetensors"),
        "FLUX (16채널)":        ("vae",  "ae.safetensors"),
        "Qwen-Image (16채널)":  ("vae",  "qwen_image_vae.safetensors")}
''')

md('''
### 셀 2 · VAE 불러오기 · ComfyUI 연결
- SD1.5 체크포인트에서 VAE 만 꺼내 쓴다(파라미터 8,400만 — 인코더 · 디코더)
- `comfy_run(그래프)` : ComfyUI 에 **API 형식 워크플로**(노드 번호 → 노드 종류 · 입력)를 보내고(`POST /prompt`) → 끝날 때까지 기록을 묻고(`GET /history`) → 결과 파일을 받는다(`GET /view`). 05 에서 자세히
''')
code('''
# 셀 2 · VAE 불러오기 · ComfyUI 연결
import io, math, time, uuid, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, requests, torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline

have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic") if f in have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False})
torch.set_grad_enabled(False)
assert torch.cuda.is_available(), "GPU 가 보이지 않음 — 5-2 는 NVIDIA GPU 필수(README 「실행 환경」)"
DEV = "cuda"
ROOT = next(p for p in (Path("/work"), Path.cwd().parents[1]) if (p / "5-1" / "images").exists())
IMAGE_DIR, DOC_DIR, OUT_DIR = ROOT / "5-1" / "images", ROOT / "5-1" / "docs", ROOT / "5-2" / "outputs" / "comfy"

pipe = StableDiffusionPipeline.from_single_file(SD_FILE, torch_dtype=torch.float32)
vae = pipe.vae.to(DEV); del pipe                       # VAE 만 float32 로(작은 값 차이를 재려고)
SF = vae.config.scaling_factor
print(f"SD1.5 VAE · 파라미터 {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M · 스케일 계수 {SF}")


def square(path, size=512):
    im = Image.open(path).convert("RGB"); s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((size, size), Image.BICUBIC)


def to_x(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(DEV)


def to_img(x):
    return Image.fromarray(((x[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))


def psnr(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return 10 * math.log10(255 ** 2 / max(((a - b) ** 2).mean(), 1e-9))


# ── ComfyUI API ──
CID = str(uuid.uuid4())
def comfy_upload(im, name):
    buf = io.BytesIO(); im.save(buf, format="PNG")
    r = requests.post(f"{COMFY_URL}/upload/image", files={"image": (name, buf.getvalue(), "image/png")}, data={"overwrite": "true"})
    r.raise_for_status(); return r.json()["name"]

def comfy_run(graph, timeout=1800):
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph, "client_id": CID})
    if r.status_code != 200: raise RuntimeError(r.text[:500])
    pid, t0 = r.json()["prompt_id"], time.time()
    while time.time() - t0 < timeout:
        h = requests.get(f"{COMFY_URL}/history/{pid}").json().get(pid)
        if h and h["status"].get("completed"): return h
        if h and h["status"].get("status_str") == "error": raise RuntimeError(h["status"]["messages"][-1])
        time.sleep(0.3)
    raise TimeoutError(pid)

def comfy_images(h):
    out = []
    for o in h["outputs"].values():
        for f in o.get("images", []):
            r = requests.get(f"{COMFY_URL}/view", params={"filename": f["filename"], "subfolder": f["subfolder"], "type": f["type"]})
            out.append(Image.open(io.BytesIO(r.content)).convert("RGB"))
    return out

ok = requests.get(f"{COMFY_URL}/system_stats", timeout=5).json()
print(f"ComfyUI {ok['system']['comfyui_version']} · {ok['devices'][0]['name']}")
''')

md('''
### 셀 3 · 압축 — 48배
- 인코더 : 그림 [1, 3, 512, 512] → latent [1, 4, 64, 64]. 가로 · 세로 각각 8분의 1, 채널 3 → 4
- 디코더 : latent → 그림. 되살린 그림과 원본의 차이를 PSNR(높을수록 같음, 40dB 이상이면 눈으로 구별 어려움)로
- U-Net 은 이 작은 latent 위에서만 일한다(01 셀 3) — VAE 는 생성 원리가 아니라 **계산을 줄이는 압축 장치**(장표 「VAE」 1번)
''')
code('''
# 셀 3 · 압축 — 48배
photo = square(IMAGE_DIR / PHOTO)
x = to_x(photo)
d = vae.encode(x).latent_dist
z = d.mean
back = to_img(vae.decode(z).sample)
rows = [("그림", tuple(x.shape), f"{x.numel():,}"), ("latent", tuple(z.shape), f"{z.numel():,}")]
print(pd.DataFrame(rows, columns=["", "모양", "값 개수"]).to_string(index=False))
print(f"압축 : {x.numel():,} / {z.numel():,} = {x.numel() / z.numel():.0f}배 · 되살린 그림 PSNR {psnr(photo, back):.1f} dB")
crop = (180, 260, 330, 410)
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
for a, im, t in zip(ax, [photo, back, photo.crop(crop), back.crop(crop)], ["원본 512×512", "VAE 왕복", "원본 확대", "왕복 확대"]):
    a.imshow(im, interpolation="nearest"); a.set_title(t, fontsize=9); a.axis("off")
plt.tight_layout(); plt.show()
''')

md('''
### 셀 4 · latent 4채널을 채널별로 보기
- 4채널은 빨강 · 초록 · 파랑 · 투명도가 아니다. VAE 가 학습하며 스스로 정한 특징 채널(장표 「Latent 채널」 1번)
- 채널마다 64×64 흑백 지도로 그린다 — 윤곽은 보이지만 채널마다 무엇을 담는지는 이름이 없다(3강 "임베딩 축에는 이름이 없다"와 같은 성격)
''')
code('''
# 셀 4 · latent 4채널을 채널별로 보기
fig, ax = plt.subplots(1, 5, figsize=(14, 3))
ax[0].imshow(photo); ax[0].set_title("원본", fontsize=9); ax[0].axis("off")
for i in range(4):
    m = z[0, i].cpu().numpy()
    ax[i + 1].imshow(m, cmap="gray"); ax[i + 1].set_title(f"채널 {i} · 범위 {m.min():.1f}~{m.max():.1f}", fontsize=9); ax[i + 1].axis("off")
plt.tight_layout(); plt.show()
''')

md('''
### 셀 5 · latent 의 크기 — 스케일 계수
- 반려동물 사진 40장을 인코드해 latent 값의 표준편차를 잰다
- 디퓨전은 latent 에 **표준편차 1 짜리 노이즈**를 섞는다(01 셀 4). latent 자체의 크기가 1 근처여야 섞는 비율 a · b 가 뜻대로 작동한다
- SD1.5 는 latent 에 **0.18215 를 곱해** 크기를 1 근처로 맞춘다 = 학습 데이터에서 잰 표준편차의 역수(LDM 논문 "성분별 표준편차로 재스케일")
- 흔한 설명 "KL 손실이 latent 를 표준 정규분포에 모음"과 비교 : KL 이 정말 그랬다면 곱할 필요가 없다 → 셀 6
''')
code('''
# 셀 5 · latent 의 크기 — 스케일 계수
meta = pd.read_csv(IMAGE_DIR / "images.csv")
Z = torch.cat([vae.encode(to_x(square(IMAGE_DIR / f))).latent_dist.mean for f in meta[meta.group == "pet"].file])
print(f"사진 {len(Z)}장 · latent {tuple(Z.shape)}")
rows = [(f"채널 {i}", f"{Z[:, i].mean():+.3f}", f"{Z[:, i].std():.3f}") for i in range(4)]
rows.append(("전체", f"{Z.mean():+.3f}", f"{Z.std():.3f}"))
print(pd.DataFrame(rows, columns=["", "평균", "표준편차"]).to_string(index=False))
print(f"\\n× 스케일 계수 {SF} → 표준편차 {(Z * SF).std():.3f}   (1 / 표준편차 = {1 / Z.std():.5f})")
''')

md('''
### 셀 6 · 인코더가 내는 μ · σ
- VAE 인코더는 latent 를 값 하나가 아니라 **분포(평균 μ · 표준편차 σ)** 로 낸다. 셀 3~5 는 μ 만 썼다
- reparameterization : `z = μ + σ · ε` (ε 는 표준 정규 노이즈) — 랜덤은 ε 에서, 학습할 값은 μ · σ 에. 직접 계산해 라이브러리 `sample()` 과 비교
- KL 손실 = 각 칸의 분포가 표준 정규분포 N(0, 1) 에서 얼마나 먼가. SD 의 VAE 는 이 손실에 **0.000001** 의 가중치만 주고 학습했다(latent-diffusion 학습 설정 `kl_weight`) → σ 는 아주 작고 μ 는 0 에서 멀다
''')
code('''
# 셀 6 · 인코더가 내는 μ · σ
mu, sd = d.mean, d.std
eps = torch.randn(mu.shape, generator=torch.Generator(DEV).manual_seed(0), device=DEV)
mine = mu + sd * eps
lib = d.sample(generator=torch.Generator(DEV).manual_seed(0))
kl = 0.5 * (mu ** 2 + sd ** 2 - 1 - torch.log(sd ** 2)).sum().item()
print(f"μ : 평균 {mu.mean():+.3f} · 표준편차 {mu.std():.3f} · 절댓값 평균 {mu.abs().mean():.3f}")
print(f"σ : 평균 {sd.mean():.5f} · 최대 {sd.max():.5f}   → σ / |μ| ≈ {sd.mean() / mu.abs().mean():.4f}")
print(f"z = μ + σ·ε 직접 계산 vs 라이브러리 sample() : 최대 차이 {(mine - lib).abs().max().item():.2e}")
a, b = to_img(vae.decode(mu).sample), to_img(vae.decode(lib).sample)
print(f"μ 로 디코드 vs 샘플로 디코드 : PSNR {psnr(a, b):.1f} dB (샘플링 랜덤이 그림에 거의 안 보임)")
print(f"이 사진 한 장의 KL (N(0,1) 과의 거리) : {kl:,.0f}  × 가중치 0.000001 = {kl * 1e-6:.4f}")
''')

md('''
### 셀 7 · latent 한 칸을 바꾸면
- latent 의 가운데 한 칸(4채널 모두)에 값을 더하고 디코드 → 그림에서 바뀐 영역을 본다
- latent 한 칸 = 그림의 8×8 블록에 대응하지만, 디코더가 주변 칸을 함께 보므로 영향은 조금 더 넓게 퍼진다
''')
code('''
# 셀 7 · latent 한 칸을 바꾸면
z2 = mu.clone(); z2[0, :, 32, 32] += 3 * mu.std()
diff = (vae.decode(z2).sample - vae.decode(mu).sample).abs().mean(1)[0].cpu().numpy()
ys, xs = np.where(diff > 0.02)
print(f"바뀐 칸 : latent (32, 32) 1칸 → 그림에서 차이 0.02 넘는 픽셀 {len(ys):,}개 · 범위 x {xs.min()}~{xs.max()} · y {ys.min()}~{ys.max()} "
      f"({xs.max() - xs.min() + 1}×{ys.max() - ys.min() + 1} 픽셀, 대응 블록은 x·y 256~263)")
fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
ax[0].imshow(to_img(vae.decode(z2).sample)); ax[0].set_title("가운데 1칸을 바꾼 latent → 그림", fontsize=9)
ax[1].imshow(diff[200:320, 200:320], cmap="magma"); ax[1].set_title("차이 (가운데 120×120 확대)", fontsize=9)
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
''')

md('''
### 셀 8 · 글자 복원 — 4채널 vs 16채널 VAE (ComfyUI)
- 5-1 문서 스크린샷을 폭 512 로 줄여(글자 높이 약 11픽셀) VAE 로 **인코드 → 디코드** 만 한다. 생성 없음 = VAE 압축 손실만 본다
- ComfyUI 그래프 : `LoadImage → VAEEncode → (SaveLatent) → VAEDecode → SaveImage`. VAE 만 바꿔 네 번
- 장표 「VAE」 4번 "작은 글자 · 날카로운 경계가 흐려짐" + 「Latent 채널」 "SD3 · FLUX 는 16채널" — 채널을 늘린 효과를 같은 문서로
''')
code('''
# 셀 8 · 글자 복원 — 4채널 vs 16채널 VAE (ComfyUI)
doc = Image.open(DOC_DIR / DOC).convert("RGB")
h = round(doc.height * DOC_WIDTH / doc.width) // 16 * 16
doc = doc.resize((DOC_WIDTH, round(doc.height * DOC_WIDTH / doc.width)), Image.LANCZOS).crop((0, 0, DOC_WIDTH, h))
name = comfy_upload(doc, "5-2_doc.png")
res, rows = {}, []
for label, (kind, f) in VAES.items():
    loader = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": f}} if kind == "ckpt" else \\
             {"class_type": "VAELoader", "inputs": {"vae_name": f}}
    vae_out = ["1", 2] if kind == "ckpt" else ["1", 0]
    tag = f"5-2/03/{Path(f).stem}"
    g = {"1": loader,
         "2": {"class_type": "LoadImage", "inputs": {"image": name}},
         "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": vae_out}},
         "4": {"class_type": "SaveLatent", "inputs": {"samples": ["3", 0], "filename_prefix": tag}},
         "5": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": vae_out}},
         "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": tag}}}
    hist = comfy_run(g)
    im = comfy_images(hist)[0]
    lat = next(o["latents"][0] for o in hist["outputs"].values() if "latents" in o)
    shape = tuple(load_file(OUT_DIR / lat["subfolder"] / lat["filename"])["latent_tensor"].shape)
    res[label] = im
    rows.append((label, f, str(shape), f"{np.prod(shape):,}", f"{psnr(doc, im):.1f}"))
print(f"문서 {DOC} → {doc.size[0]}×{doc.size[1]} (값 {doc.size[0] * doc.size[1] * 3:,}개)")
print(pd.DataFrame(rows, columns=["VAE", "파일", "latent 모양", "값 개수", "PSNR dB"]).to_string(index=False))
box = (0, 150, 256, 250)
fig, ax = plt.subplots(len(res) + 1, 1, figsize=(9, 2.3 * (len(res) + 1)))
for a, (t, im) in zip(ax, [("원본", doc)] + list(res.items())):
    a.imshow(im.crop(box).resize(((box[2] - box[0]) * 3, (box[3] - box[1]) * 3), Image.NEAREST)); a.set_title(t, fontsize=9); a.axis("off")
plt.tight_layout(); plt.show()
''')

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT, len(cells), 'cells')
