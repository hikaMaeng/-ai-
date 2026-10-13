"""6-1 노트북 공통 함수 — 데이터 받기 · 들려 주기 · 그림 · 글자 오류율(CER)

노트북마다 셀 2 에서 `from voice_common import *` 로 불러 쓴다.
"""
import io
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import editdistance
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import Audio, display

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # 6-1/
DATA = ROOT / "data"                    # 수강생 녹음 · 노트북이 만든 소리(.gitignore)
RESULTS = HERE / "results"              # 노트북 출력 그림 · 표(.gitignore)
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic") if f in _have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False})

SR = 16000                              # 이 실습의 기본 샘플링레이트(Whisper · Qwen3-ASR 입력)


# ── 데이터 : FLEURS 한국어 test (Google, CC BY 4.0) ─────────────────────────
#   382줄 = 문장 270개를 화자 여럿이 읽음(같은 문장 다른 화자 쌍이 있음)
#   hfcache 볼륨에 한 번 받으면(291MB) 다시 받지 않음
def fleurs_ko():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("google/fleurs", "ko_kr/test/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
    df = pd.read_parquet(p)
    df["gender"] = df["gender"].map({0: "남", 1: "여"})
    df["sec"] = df["num_samples"] / SR
    return df


def wav_of(df, i, sr=SR):
    """FLEURS 한 줄 → float32 파형(1차원). sr 이 16k 가 아니면 다시 샘플링"""
    x, s = sf.read(io.BytesIO(df.iloc[i]["audio"]["bytes"]), dtype="float32")
    return librosa.resample(x, orig_sr=s, target_sr=sr) if s != sr else x


def load_wav(path, sr=SR):
    x, s = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return librosa.resample(x, orig_sr=s, target_sr=sr) if s != sr else x


def play(x, sr=SR, label=None):
    if label:
        print(label)
    display(Audio(np.asarray(x, dtype=np.float32), rate=sr))


# ── 오류율 ─────────────────────────────────────────────────────────────────
def cer(hyp, ref):
    """글자 오류율 = 띄어쓰기를 뺀 글자 편집거리 ÷ 정답 글자 수(한국어 ASR 의 기본 지표)"""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    return editdistance.eval(h, r) / max(1, len(r))


def wer(hyp, ref):
    """단어 오류율 = 단어 편집거리 ÷ 정답 단어 수(영어 · 띄어쓰기 단위)"""
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c.isspace()).split()
    h, r = norm(hyp), norm(ref)
    return editdistance.eval(h, r) / max(1, len(r))


def strip_punct(s):
    return "".join(c for c in s if c.isalnum() or c.isspace())


# ── GPU 메모리 ─────────────────────────────────────────────────────────────
def gpu_free():
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_peak_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0
