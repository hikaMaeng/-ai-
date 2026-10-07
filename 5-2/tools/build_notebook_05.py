"""5-2 실습 노트북 05 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook_05.py   → ../notebooks/05_ComfyUI로_최신모델.ipynb (출력 없이)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / '05_ComfyUI로_최신모델.ipynb'
cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': s.strip('\n').splitlines(keepends=True)})


md('''
# 실습 05 · ComfyUI 로 최신 모델 — Jupyter 가 ComfyUI 를 부리기

01~04 에서 부품을 열어 본 것을 **실제 최신 모델 셋**(FLUX.1-dev · Qwen-Image · Z-Image)과 SD1.5 로 돌린다. 생성은 ComfyUI 가 하고, 이 노트북은 **워크플로를 만들어 보내고 결과를 받아** 비교한다. ComfyUI 화면(http://localhost:8188)은 이 노트북이 저장한 워크플로를 열어 보거나 직접 돌려 볼 때만 쓴다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | ComfyUI 연결 · API 세 단계 · 워크플로 만드는 함수 | 실습 장표 「Jupyter가 ComfyUI를 부리는 법」 |
| 3 | 워크플로 = 부품 목록 — 모델 넷의 노드를 강의 부품에 대응 · 화면용 저장 | 실습 장표 「05 구성 · 셀 3 워크플로 = 부품」 · 강의 「CFG (4) — ComfyUI 화면에서」 |
| 4 | SD1.5 기본형 — U-Net 디퓨전 | 파이프라인 도해 |
| 5 | 세 모델 같은 프롬프트 — 신문 헤드라인 글자 | DiT 텐서 흐름 · Qwen-Image 분석 3번 |
| 6 | 단계별 완성본 추정 — DiT 에서도 구도가 먼저인가 | U-Net과 타임스텝 4~7번 |
| 7 | CFG 와 가이던스 — Qwen-Image cfg · FLUX 가이던스 증류 | CFG (3) · (4) |
| 8 | 스텝 수 · 샘플러 — FLUX dev vs schnell(4스텝 증류) · Z-Image 샘플러 셋 | sampler · 플로 매칭 — 스텝 수와 시프트 |
| 9 | 시프트 — 노이즈 큰 구간에 스텝 몰기 | 플로 매칭 — 스텝 수와 시프트 · 플로 매칭 — ComfyUI 화면 |
| 10 | 한국어 프롬프트 · 한글 간판 — 조건 인코더 셋 비교 | Qwen-Image 분석 3번 · 5-1 01 셀 4 |
| 11 | 조건 인코더 바꿔치기 — Z-Image 에 다른 LLM | Diffusion 이미지 생성 1번 · U-Net이 프롬프트를 반영하는 방법 4번 |
| 12 | 내 프롬프트 | – |

- 실행 : **Run → Run All Cells**. GPU 필수(VRAM 16GB 이상 권장 — ComfyUI 가 모델을 번갈아 올림). 전체 약 45분(RTX 4080 실측 44분) — 그림 한 장 약 2초(SD1.5)~5분(Qwen-Image 첫 로드 포함)
- 모델 : `../4-1/comfyui/models` (README 「모델 받기」). ComfyUI 는 같은 그래프 · 같은 값이면 결과를 다시 계산하지 않는다(캐시) → 시간은 처음 실행 기준
''')

md('''
### 셀 1 · 설정
- `MODELS` : 모델별 부품 파일과 권장 설정(05 셀 3 표). 강사 PC 의 원래 워크플로(ComfyUI 화면)와 같은 값
- `PROMPT` : 셀 5 · 6 의 공통 프롬프트 — 신문 헤드라인 글자를 넣어 글자 렌더링을 함께 본다
''')
code('''
# 셀 1 · 설정
import os
COMFY_URL = os.environ.get("COMFY_URL", "http://comfy:8188")
SEED   = 42
PROMPT = ('A photorealistic close-up of a calico cat wearing tiny round glasses, reading a newspaper '
          'with the headline "NEURON DEV", warm cafe lighting, ultra detailed')
KO_PROMPT = '한복을 입은 고양이가 "뉴런데브" 라고 크게 쓴 나무 간판 앞에 앉아 있는 사진, 따뜻한 햇빛'
MY_PROMPT = "an engineer inspecting a chemical reactor in a clean factory, documentary photo"
MY_MODEL  = "Z-Image"

MODELS = {
    "SD1.5":      dict(ckpt="v1-5-pruned-emaonly.safetensors", size=512, steps=20, cfg=7.5, sampler="ddim", scheduler="ddim_uniform"),
    "FLUX.1-dev": dict(unet=("UnetLoaderGGUF", "flux1-dev-Q4_K_S.gguf"),
                       clip=("DualCLIPLoaderGGUF", "t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux"),
                       vae="ae.safetensors", size=1024, steps=20, cfg=1.0, guidance=3.5, sampler="euler", scheduler="simple"),
    "Qwen-Image": dict(unet=("UnetLoaderGGUF", "qwen-image-Q4_K_M.gguf"),
                       clip=("CLIPLoaderGGUF", "Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf", "qwen_image"),
                       vae="qwen_image_vae.safetensors", size=1328, steps=20, cfg=2.5, shift=3.1, sampler="euler", scheduler="simple",
                       negative="blurry, low quality, distorted, watermark, text artifacts"),
    "Z-Image":    dict(unet=("UNETLoader", "z_image_bf16.safetensors"),
                       clip=("CLIPLoader", "qwen_3_4b.safetensors", "lumina2"),
                       vae="z_image_ae.safetensors", size=1024, steps=30, cfg=4.0, shift=3.0, sampler="res_multistep", scheduler="simple"),
}
SWEEP_SIZE = 1024     # 셀 6~11 의 비교 실험 해상도(Qwen-Image 도 1024 로 줄여 시간을 아낌)
''')

md('''
### 셀 2 · ComfyUI 연결 · API 세 단계
- ComfyUI 는 웹 서버다. 화면도 이 API 를 쓴다
  1. `POST /prompt` — **API 형식 워크플로**(노드 번호 → `class_type` · `inputs`)를 보냄 → 작업 번호
  2. `GET /history/{작업 번호}` — 끝났는지 · 결과 파일 이름 · 실행 기록
  3. `GET /view?filename=…` — 결과 그림 받기
- 노드의 입력은 값(숫자 · 글) 또는 **다른 노드의 출력** `["노드 번호", 출력 번호]` — 화면의 선 하나가 이 한 줄
- `graph(모델, …)` : 셀 1 의 설정으로 워크플로를 만든다. `run(…)` : 보내고 기다려 그림과 실행 시간(ComfyUI 기록 기준)을 돌려준다
- `free()` : ComfyUI 가 올려 둔 모델을 GPU 에서 내린다(`POST /free`) — GPU 한 장을 Jupyter 와 나눠 쓸 때
''')
code('''
# 셀 2 · ComfyUI 연결 · API 세 단계
import io, json, math, time, uuid, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, requests
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic") if f in have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False})
ROOT = next(p for p in (Path("/work"), Path.cwd().parents[1]) if (p / "5-2").exists())
WF_DIR = ROOT / "5-2" / "workflows"
CID = str(uuid.uuid4())

stats = requests.get(f"{COMFY_URL}/system_stats", timeout=5).json()
oi = requests.get(f"{COMFY_URL}/object_info").json()
dev = stats["devices"][0]
print(f"ComfyUI {stats['system']['comfyui_version']} · {dev['name']} · VRAM {dev['vram_total'] / 2**30:.0f}GB · 노드 종류 {len(oi)}개")


def node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def graph(model, prompt, negative=None, seed=SEED, steps=None, cfg=None, sampler=None, scheduler=None, size=None,
          shift="model", guidance="model", stop_at=None, clip=None, prefix="5-2/05/x"):
    """셀 1 의 MODELS 설정으로 API 형식 워크플로(dict)를 만든다. 인자를 주면 그 값만 바꿈"""
    m = MODELS[model]
    steps, cfg = steps or m["steps"], m["cfg"] if cfg is None else cfg
    sampler, scheduler, size = sampler or m["sampler"], scheduler or m["scheduler"], size or m["size"]
    negative = m.get("negative", "") if negative is None else negative
    g = {}
    if "ckpt" in m:                                                    # SD1.5 : 한 파일에 U-Net · 글 인코더 · VAE
        g["1"] = node("CheckpointLoaderSimple", ckpt_name=m["ckpt"])
        mdl, clp, vae = ["1", 0], ["1", 1], ["1", 2]
        g["5"] = node("EmptyLatentImage", width=size, height=size, batch_size=1)
    else:
        u = m["unet"]
        g["1"] = node(u[0], unet_name=u[1], **({"weight_dtype": "default"} if u[0] == "UNETLoader" else {}))
        c = clip or m["clip"]
        g["2"] = node(c[0], clip_name1=c[1], clip_name2=c[2], type=c[3]) if c[0].startswith("Dual") else node(c[0], clip_name=c[1], type=c[2])
        g["3"] = node("VAELoader", vae_name=m["vae"])
        mdl, clp, vae = ["1", 0], ["2", 0], ["3", 0]
        s = m.get("shift") if shift == "model" else shift
        if s:
            g["4"] = node("ModelSamplingAuraFlow", model=mdl, shift=s); mdl = ["4", 0]
        g["5"] = node("EmptySD3LatentImage", width=size, height=size, batch_size=1)
    g["6"] = node("CLIPTextEncode", text=prompt, clip=clp); pos = ["6", 0]
    g["7"] = node("CLIPTextEncode", text=negative, clip=clp)
    gd = m.get("guidance") if guidance == "model" else guidance
    if gd is not None:
        g["8"] = node("FluxGuidance", conditioning=pos, guidance=gd); pos = ["8", 0]
    if stop_at is None:
        g["9"] = node("KSampler", model=mdl, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler,
                      positive=pos, negative=["7", 0], latent_image=["5", 0], denoise=1.0)
    else:                                                              # stop_at 단계에서 멈추고 그때의 완성본 추정을 꺼냄
        g["9"] = node("KSamplerAdvanced", model=mdl, add_noise="enable", noise_seed=seed, steps=steps, cfg=cfg,
                      sampler_name=sampler, scheduler=scheduler, positive=pos, negative=["7", 0], latent_image=["5", 0],
                      start_at_step=0, end_at_step=stop_at, return_with_leftover_noise="disable")
    g["10"] = node("VAEDecode", samples=["9", 0], vae=vae)
    g["11"] = node("SaveImage", images=["10", 0], filename_prefix=prefix)
    return g


def run(g, timeout=3600):
    """워크플로를 보내고(1) 끝날 때까지 기록을 묻고(2) 그림을 받는다(3). 시간은 ComfyUI 실행 기록의 시작~끝"""
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": g, "client_id": CID})
    if r.status_code != 200:
        raise RuntimeError(r.json().get("error", r.text)) from None
    pid, t0 = r.json()["prompt_id"], time.time()
    while time.time() - t0 < timeout:
        h = requests.get(f"{COMFY_URL}/history/{pid}").json().get(pid)
        if h and h["status"].get("status_str") == "error":
            msg = [m for m in h["status"]["messages"] if m[0] == "execution_error"]
            raise RuntimeError(msg[0][1]["exception_message"].strip()[:300] if msg else h["status"])
        if h and h["status"].get("completed"):
            ts = {m[0]: m[1].get("timestamp") for m in h["status"]["messages"]}
            sec = (ts["execution_success"] - ts["execution_start"]) / 1000
            ims = []
            for o in h["outputs"].values():
                for f in o.get("images", []):
                    b = requests.get(f"{COMFY_URL}/view", params={"filename": f["filename"], "subfolder": f["subfolder"], "type": f["type"]}).content
                    ims.append(Image.open(io.BytesIO(b)).convert("RGB"))
            return ims[0], sec
        time.sleep(0.5)
    raise TimeoutError(pid)


def free():
    requests.post(f"{COMFY_URL}/free", json={"unload_models": True, "free_memory": True})


def show(ims, titles=None, cols=None, size=3.0, suptitle=None):
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
### 셀 3 · 워크플로 = 부품 목록
- 모델 넷의 워크플로를 만들어 노드를 **강의의 부품**에 대응시킨다. 노드 이름이 낯설 뿐, 전부 01~04 에서 연 부품이다
  - `UnetLoaderGGUF` · `UNETLoader` : 이름은 U-Net 이지만 FLUX · Qwen-Image · Z-Image 는 **DiT** 를 불러온다(ComfyUI 가 예전 이름을 그대로 씀)
  - GGUF : 웨이트를 4 · 5비트로 줄여 저장한 파일 형식(5-1 의 양자화와 같은 발상) — 12B FLUX 가 약 6.8GB
- 저장 : 이 워크플로들을 `5-2/workflows/` 에 두 형식으로 저장한다
  - `*_api.json` : API 형식(이 노트북이 보내는 그대로)
  - `*_t2i.json` : 화면 형식(노드 위치 · 선 포함, `tools/api_to_ui.py` 로 변환) → ComfyUI 화면의 워크플로 목록 「5-2」 에서 그래프로 열린다. 화면에서 「실행」 하면 같은 그림
''')
code('''
# 셀 3 · 워크플로 = 부품 목록
ROLE = {"CheckpointLoaderSimple": "디노이저 U-Net + 글 인코더 + VAE (한 파일)", "UnetLoaderGGUF": "디노이저 (DiT, 4비트 GGUF)",
        "UNETLoader": "디노이저 (DiT)", "DualCLIPLoaderGGUF": "조건 인코더 둘 (T5 토큰 + CLIP 풀링)",
        "CLIPLoaderGGUF": "조건 인코더 (VLM)", "CLIPLoader": "조건 인코더 (LLM)", "VAELoader": "VAE",
        "ModelSamplingAuraFlow": "시프트 (노이즈 수준 간격)", "EmptyLatentImage": "시작 latent 4채널", "EmptySD3LatentImage": "시작 latent 16채널",
        "CLIPTextEncode": "프롬프트 → 조건 c", "FluxGuidance": "가이던스 값 (증류된 CFG)", "KSampler": "디노이즈 루프 (샘플러 · 스케줄러 · CFG · 시드)",
        "VAEDecode": "VAE 디코더", "SaveImage": "저장"}
import sys; sys.path.insert(0, str(ROOT / "5-2" / "tools"))
from api_to_ui import to_ui                     # API 형식 → 화면 형식(노드 위치 · 선). 화면은 API 형식 파일을 그래프로 열지 못함
tab = {}
for name in MODELS:
    g = graph(name, PROMPT, prefix=f"5-2/05/{name}")
    (WF_DIR / f"{name}_t2i_api.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
    (WF_DIR / f"{name}_t2i.json").write_text(json.dumps(to_ui(g, oi), ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in g.items():
        key = ROLE[v["class_type"]]
        tab.setdefault(key, {})[name] = v["class_type"]
order = list(dict.fromkeys(ROLE.values()))
T = pd.DataFrame([{**{"부품": r}, **tab[r]} for r in order if r in tab]).fillna("–")
print(T.to_string(index=False))
print(f"\\n저장 : 화면용 {sorted(p.name for p in WF_DIR.glob('*_t2i.json'))} · API 용 *_api.json")
print("\\n설정값 :")
print(pd.DataFrame([(n, m["size"], m["steps"], m["cfg"], m.get("guidance", "–"), m.get("shift", "–"), m["sampler"], m["scheduler"])
                    for n, m in MODELS.items()], columns=["모델", "크기", "스텝", "cfg", "가이던스", "시프트", "샘플러", "스케줄러"]).to_string(index=False))
''')

md('''
### 셀 4 · SD1.5 기본형
- 01 과 같은 프롬프트 · 시드 · DDIM 20단계 · cfg 7.5 를 ComfyUI 로. 그림은 01 과 다르다 — ComfyUI 는 시작 노이즈를 CPU 에서 만들고 diffusers 는 GPU 에서 만들어 **같은 시드라도 노이즈 수열이 다름**
- 노드 11개 = 01 셀 3 의 ①~⑥
''')
code('''
# 셀 4 · SD1.5 기본형
im, sec = run(graph("SD1.5", "a red car on the beach", prefix="5-2/05/sd15"))
print(f"SD1.5 · 512×512 · 20단계 · {sec:.1f}초 (모델 올리기 포함, 다시 돌리면 캐시)")
im2, sec2 = run(graph("SD1.5", "a red car on the beach", seed=SEED + 1, prefix="5-2/05/sd15"))
print(f"시드만 바꿔 다시 : {sec2:.1f}초 (모델이 올라가 있으면 생성만)")
show([im, im2], [f"seed {SEED}", f"seed {SEED + 1}"], size=3.4)
''')

md('''
### 셀 5 · 세 모델 같은 프롬프트
- 같은 프롬프트(헤드라인 "NEURON DEV") · 같은 시드, 각 모델의 권장 설정(셀 3 표)
- 볼 것 : ① 헤드라인 글자를 정확히 썼나 ② 고양이 · 안경 · 신문 · 카페 조명을 모두 반영했나 ③ 시간
- 장표 「Qwen-Image 분석」 3번 "조건 c 가 풍부해져 긴 텍스트 렌더링에 유리" · 「Latent 채널」 16채널(03 셀 8)
''')
code('''
# 셀 5 · 세 모델 같은 프롬프트
res = {}
for name in ["FLUX.1-dev", "Qwen-Image", "Z-Image"]:
    res[name] = run(graph(name, PROMPT, prefix=f"5-2/05/{name}"))
    print(f"{name:11s} · {MODELS[name]['size']}² · {MODELS[name]['steps']}단계 · {res[name][1]:.1f}초")
show([v[0] for v in res.values()], [f"{k} · {v[1]:.0f}초" for k, v in res.items()], size=4.2)
''')

md('''
### 셀 6 · 단계별 완성본 추정 — DiT 에서도 구도가 먼저인가
- Z-Image 를 30단계로 돌리되 k 단계에서 멈추고 **그때의 완성본 추정**을 디코드한다(`KSamplerAdvanced` 의 `end_at_step` · 남은 노이즈 없이)
- 01 셀 6(SD1.5 U-Net)과 같은 실험을 DiT · 플로 매칭 모델로 — "초반 구도, 후반 디테일"이 구조와 상관없이 나타나는가
''')
code('''
# 셀 6 · 단계별 완성본 추정
K = [1, 2, 4, 8, 15, 30]
ims, titles = [], []
for k in K:
    im, sec = run(graph("Z-Image", PROMPT, size=SWEEP_SIZE, stop_at=k, prefix="5-2/05/steps"))
    ims.append(im); titles.append(f"{k} / 30 단계")
show(ims, titles, cols=len(K), size=2.6, suptitle="Z-Image · k 단계에서 멈춘 완성본 추정")
''')

md('''
### 셀 7 · CFG 와 가이던스
- Qwen-Image : cfg 1 · 2.5(권장) · 5 · 8 — 02 셀 9 와 같은 CFG. cfg > 1 이면 조건 있는 · 없는 예측 두 번
- FLUX.1-dev : **cfg 는 1 로 고정**하고 `FluxGuidance` 값만 바꾼다(1 · 3.5 · 7) — CFG 를 학습 때 모델에 녹여 넣은 **가이던스 증류**. 예측 한 번에 CFG 효과 → 부정 프롬프트는 쓰이지 않음
- 비교 : FLUX 에 cfg 3.5 를 주면(증류 모델에 CFG 를 또 하면)
''')
code('''
# 셀 7 · CFG 와 가이던스
rows, ims, titles = [], [], []
for c in [1.0, 2.5, 5.0, 8.0]:
    im, sec = run(graph("Qwen-Image", PROMPT, cfg=c, size=SWEEP_SIZE, prefix="5-2/05/cfg"))
    ims.append(im); titles.append(f"Qwen-Image cfg {c} · {sec:.0f}초")
for gd, c in [(1.0, 1.0), (3.5, 1.0), (7.0, 1.0), (3.5, 3.5)]:
    im, sec = run(graph("FLUX.1-dev", PROMPT, cfg=c, guidance=gd, size=SWEEP_SIZE, prefix="5-2/05/cfg"))
    ims.append(im); titles.append(f"FLUX 가이던스 {gd} · cfg {c} · {sec:.0f}초")
show(ims, titles, cols=4, size=3.2)
''')

md('''
### 셀 8 · 스텝 수 · 샘플러
- FLUX.1-dev 4 · 8 · 20단계 vs **FLUX.1-schnell 4단계** — schnell 은 적은 스텝으로 끝내도록 따로 학습(증류)한 모델. 같은 구조(04 셀 4), 다른 학습
- Z-Image 30단계에서 샘플러만 셋 : euler(1차) · res_multistep(권장, 고차) · euler_ancestral(매 단계 노이즈 재주입) — 01 셀 8 과 같은 비교
''')
code('''
# 셀 8 · 스텝 수 · 샘플러
MODELS["FLUX.1-schnell"] = {**MODELS["FLUX.1-dev"], "unet": ("UnetLoaderGGUF", "flux1-schnell-Q4_K_S.gguf"), "steps": 4, "guidance": None}
ims, titles = [], []
for name, st in [("FLUX.1-dev", 4), ("FLUX.1-dev", 8), ("FLUX.1-dev", 20), ("FLUX.1-schnell", 4)]:
    im, sec = run(graph(name, PROMPT, steps=st, size=SWEEP_SIZE, prefix="5-2/05/steps"))
    ims.append(im); titles.append(f"{name} · {st}단계 · {sec:.0f}초")
for smp in ["euler", "res_multistep", "euler_ancestral"]:
    im, sec = run(graph("Z-Image", PROMPT, sampler=smp, size=SWEEP_SIZE, prefix="5-2/05/sampler"))
    ims.append(im); titles.append(f"Z-Image · {smp} · {sec:.0f}초")
show(ims, titles, cols=4, size=3.2)
''')

md('''
### 셀 9 · 시프트
- Z-Image 시프트 1 · 3(권장) · 6 — 04 셀 7 의 곡선을 실제 모델로. 시프트가 작으면 노이즈 큰 구간(구도를 정하는 단계)에 스텝이 적게 배정된다
''')
code('''
# 셀 9 · 시프트
ims, titles = [], []
for s in [1.0, 3.0, 6.0]:
    im, sec = run(graph("Z-Image", PROMPT, shift=s, size=SWEEP_SIZE, prefix="5-2/05/shift"))
    ims.append(im); titles.append(f"Z-Image · shift {s}")
show(ims, titles, size=3.6)
''')

md('''
### 셀 10 · 한국어 프롬프트 · 한글 간판
- 같은 한국어 프롬프트를 조건 인코더가 다른 세 모델에 — FLUX(T5-XXL + CLIP-L, 영어 위주) · Qwen-Image(Qwen2.5-VL-7B) · Z-Image(Qwen3-4B)
- 볼 것 : ① 한복 · 고양이 · 나무 간판을 알아들었나(이해) ② "뉴런데브" 한글을 썼나(렌더링 — 디노이저와 VAE 의 몫도 있음)
- 5-1 01 셀 4 : CLIP 토크나이저는 한글 5글자를 10조각으로 쪼갰다 — 조건 인코더를 LLM · VLM 으로 바꾸는 이유
''')
code('''
# 셀 10 · 한국어 프롬프트 · 한글 간판
ims, titles = [], []
for name in ["FLUX.1-dev", "Qwen-Image", "Z-Image"]:
    im, sec = run(graph(name, KO_PROMPT, size=SWEEP_SIZE, prefix="5-2/05/korean"))
    ims.append(im); titles.append(f"{name} · {sec:.0f}초")
print(KO_PROMPT)
show(ims, titles, size=4.0)
''')

md('''
### 셀 11 · 조건 인코더 바꿔치기
- Z-Image 의 조건 인코더 Qwen3-4B 를 **다른 LLM 두 개**로 바꿔 넣는다 — Qwen3.5-4B(같은 회사의 다음 판) · Qwen3.5-2B
- 04 셀 4 : Z-Image 의 글 투영 입력 폭은 2560 = Qwen3-4B 의 출력 폭. 폭이 다르면 행렬 곱이 안 되고(02 셀 10 ③), 폭이 같아도 좌표가 다르면 못 읽는다(02 셀 10 ②)
''')
code('''
# 셀 11 · 조건 인코더 바꿔치기
rows, ims, titles = [], [], []
for label, clip in [("Qwen3-4B (원래)", ("CLIPLoader", "qwen_3_4b.safetensors", "lumina2")),
                    ("Qwen3.5-4B", ("CLIPLoader", "qwen3.5_4b_bf16.safetensors", "lumina2")),
                    ("Qwen3.5-2B", ("CLIPLoader", "qwen3.5_2b_bf16.safetensors", "lumina2"))]:
    try:
        im, sec = run(graph("Z-Image", PROMPT, clip=clip, size=SWEEP_SIZE, prefix="5-2/05/encoder"))
        ims.append(im); titles.append(label); rows.append((label, "생성됨", f"{sec:.0f}초"))
    except Exception as e:
        rows.append((label, "오류", str(e).splitlines()[0][:110]))
print(pd.DataFrame(rows, columns=["조건 인코더", "결과", "시간 / 메시지"]).to_string(index=False))
if ims: show(ims, titles, size=3.6)
''')

md('''
### 셀 12 · 내 프롬프트
- 셀 1 의 `MY_PROMPT` · `MY_MODEL` 을 바꿔 실행. 같은 그래프는 ComfyUI 화면(워크플로 목록 「5-2」)에서도 열어 돌릴 수 있다
''')
code('''
# 셀 12 · 내 프롬프트
free()                                     # 앞 셀에서 올린 모델을 내리고 시작(GPU 한 장일 때 유리)
im, sec = run(graph(MY_MODEL, MY_PROMPT, prefix="5-2/05/mine"))
print(f"{MY_MODEL} · {sec:.1f}초")
show([im], [MY_PROMPT[:60]], size=6)
''')

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT, len(cells), 'cells')
