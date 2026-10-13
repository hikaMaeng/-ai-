"""6-2 노트북 공통 코드 — 시드 · 장치 · 그림 글꼴 · 결과 저장 · 냉각 밸브 시뮬레이터

노트북마다 첫 코드 셀에서 `from wm_common import *` 로 불러 쓴다.
- 01 : 밸브 시뮬레이터(ValveSim) · 월드모델 입력 만들기 · 개루프 예측
- 02, 03 : 시드 · 장치 · 결과 저장만 쓴다
"""
import copy
import json
import logging
import math
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"              # 노트북이 계산한 핵심 숫자(measured_0N.json)
RESULTS.mkdir(exist_ok=True)

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic") if f in _have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 100})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def logaxis(ax, x=False, y=True):
    """로그 눈금을 지수 글자(10⁻¹ 같은 모양)가 아닌 보통 숫자(0.1)로 표시 — 글꼴에 없는 기호로 눈금이 깨지는 것을 막는다"""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for on, setscale, axis in ((x, ax.set_xscale, ax.xaxis), (y, ax.set_yscale, ax.yaxis)):
        if on:
            setscale("log")
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); axis.set_minor_formatter(NullFormatter())


def seed_all(s):
    """numpy · torch 시드를 한 번에 고정"""
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def ms(xs, d=2):
    """평균 ± 표준편차 글자(표준편차는 np.std 기본값 = 모집단 표준편차)"""
    xs = np.asarray(xs, dtype=float)
    return f"{xs.mean():.{d}f} ± {xs.std():.{d}f}"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(type(o))


def save_measured(name, d):
    """핵심 숫자를 results/measured_<name>.json 으로 저장 — 장표에 옮길 숫자와 그 설정 값"""
    p = RESULTS / f"measured_{name}.json"
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    print("저장 :", p.name)


def device_info():
    if torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"


# ═══════════════════════════════════════════════════════════════════════════
#  냉각 밸브 시뮬레이터 (노트북 01)
#   상태 : 온도 T(℃) — 열 부하가 데우고, 밸브가 식힌다
#   지령 : 밸브 개도 c(0~100%). 행동 a = 개도 변화(−10~+10 %/스텝)  →  c_{t+1} = clip(c_t + a_t)
#   지연 : 스텝 t 의 온도 변화는 한 스텝 전 지령 c_{t-1} 로 식힌다.
#          지령 c_s 는 관측에 바로 보이지만 온도에는 s+2 시점부터 반영 — 개도를 바꿔도 냉각이 2스텝 뒤에 반영
#   식  : T_{t+1} = T_t + 0.1·L_t − 0.004·c_{t-1}·(T_t − 25)       (25℃ = 냉각수 온도)
#   열 부하 L_t = 기준 + 진폭·sin(2π t/주기 + 위상) + 잡음(표준편차 2)   ← 관측에 안 보임(숨은 변수)
#   관측 o_t = (T_t + 잡음(표준편차 0.5), c_t)
# ═══════════════════════════════════════════════════════════════════════════
T_TARGET, T_LIMIT = 70.0, 85.0
A_MAX = 10.0
GAIN_LOAD, K_COOL, T_WATER = 0.1, 0.004, 25.0
LOAD_NOISE, OBS_NOISE = 2.0, 0.5
EP_LEN = 80
W_ACT, W_LIMIT = 0.05, 20.0

VALVE_CONFIG = dict(T_TARGET=T_TARGET, T_LIMIT=T_LIMIT, A_MAX=A_MAX, GAIN_LOAD=GAIN_LOAD, K_COOL=K_COOL, T_WATER=T_WATER,
                    LOAD_NOISE=LOAD_NOISE, OBS_NOISE=OBS_NOISE, DELAY="지령 c_s 는 온도 T_{s+2} 부터 반영", EP_LEN=EP_LEN,
                    비용="(T-70)^2 + 0.05*a^2 + 20*max(0,T-85)^2")


def step_cost(T, a):
    """한 스텝 비용 = (T−70)² + 0.05·(개도 변화)² + 20·max(0, T−85)²  (T, a 는 numpy · torch 모두 가능)"""
    if isinstance(T, torch.Tensor):
        return (T - T_TARGET) ** 2 + W_ACT * a ** 2 + W_LIMIT * torch.relu(T - T_LIMIT) ** 2
    return (T - T_TARGET) ** 2 + W_ACT * a ** 2 + W_LIMIT * np.maximum(0.0, T - T_LIMIT) ** 2


class ValveSim:
    """n 개의 밸브를 한꺼번에 시뮬레이션(numpy). load_base · load_amp 는 (최소, 최대) 범위에서 환경마다 뽑는다"""

    def __init__(self, n, seed, T0=(50, 95), c0=(20, 80), load_base=(80, 100), load_amp=(10, 30), noise_seed=None):
        r = np.random.default_rng(seed)
        self.n = n
        self.T = r.uniform(*T0, n)
        self.c = r.uniform(*c0, n)
        self.c_prev = self.c.copy()                      # 시작 때는 지령이 안 바뀐 상태
        self.base = r.uniform(*load_base, n)
        self.amp = r.uniform(*load_amp, n)
        self.period = r.uniform(40, 100, n)
        self.phase = r.uniform(0, 2 * np.pi, n)
        self.t = 0
        self.rng = np.random.default_rng(seed + 10_000 if noise_seed is None else noise_seed)

    def load(self):
        return self.base + self.amp * np.sin(2 * np.pi * self.t / self.period + self.phase)

    def obs(self):
        """(잡음 낀 온도, 현재 지령값)"""
        return self.T + self.rng.normal(0, OBS_NOISE, self.n), self.c.copy()

    def step(self, a):
        a = np.clip(a, -A_MAX, A_MAX)
        L = self.load() + self.rng.normal(0, LOAD_NOISE, self.n)
        self.T = self.T + GAIN_LOAD * L - K_COOL * self.c_prev * (self.T - T_WATER)
        self.c_prev = self.c
        self.c = np.clip(self.c + a, 0, 100)
        self.t += 1
        return a


def collect_random(n_env, steps, seed, hold=3, keep_sims=False, **sim_kw):
    """무작위 행동 데이터. 행동은 hold 스텝마다 새로 뽑는다(너무 떨리지 않게). 반환 : dict(T_obs, c, a, T_next)  각 (n_env, steps)
    keep_sims=True 이면 각 스텝 직전의 시뮬레이터 사본도 d["sims"] 에 둔다(같은 상태에서 다시 돌려 보기 위해)"""
    sim = ValveSim(n_env, seed, **sim_kw)
    sims = []
    r = np.random.default_rng(seed + 1)
    To, C, A, Tn = [], [], [], []
    a = np.zeros(n_env)
    for t in range(steps):
        if t % hold == 0:           # 무작위 행동 + 개도가 0·100 끝에 붙어 온도가 폭주하지 않게 가운데(50%)로 약하게 당김
            a = np.clip(r.uniform(-8, 8, n_env) + 0.2 * (50 - sim.c), -A_MAX, A_MAX)
        if keep_sims:
            sims.append(copy.deepcopy(sim))
        o_T, o_c = sim.obs()
        a_eff = np.clip(a, -A_MAX, A_MAX)
        sim.step(a_eff)
        To.append(o_T); C.append(o_c); A.append(a_eff); Tn.append(sim.T.copy())
    d = {k: np.stack(v, 1) for k, v in dict(T_obs=To, c=C, a=A, T_next=Tn).items()}
    if keep_sims:
        d["sims"] = sims
    return d


def make_windows(d, k):
    """데이터 → 월드모델 입력(최근 k개 관측 + 행동)과 정답.
    입력 열 : [T_{t-k+1..t} 정규화, c_{t-k+1..t} 정규화, a_t 정규화]   정답 : T_{t+1} − T_t(관측)  (÷3)
    시작 부분은 첫 관측을 반복해서 채운다"""
    n, S = d["T_obs"].shape
    idx = np.clip(np.arange(S)[:, None] - np.arange(k - 1, -1, -1)[None, :], 0, None)       # (S, k)
    Th = d["T_obs"][:, idx]                                                                     # (n, S, k)
    Ch = d["c"][:, idx]
    x = np.concatenate([(Th - T_TARGET) / 10, Ch / 50 - 1, d["a"][..., None] / A_MAX], -1)
    y = (d["T_next"] - d["T_obs"])[..., None] / 3
    return x.reshape(-1, 2 * k + 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def mlp(i, o, h=128, layers=2):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*mods, nn.Linear(d, o))


@torch.no_grad()
def rollout(model, Th, Ch, A):
    """월드모델만으로 H 스텝 앞을 예측(개루프).
    Th, Ch : (N, k) 최근 k개 관측의 온도 · 지령 / A : (N, H) 행동열(개도 변화)  →  예측 온도 (N, H)  [T_{t+1} … T_{t+H}]
    예측한 온도를 다음 스텝의 입력으로 되먹임 — 이 되먹임이 누적 오차의 원인"""
    Th, Ch = Th.clone(), Ch.clone()
    out = []
    for h in range(A.shape[1]):
        a = A[:, h].clamp(-A_MAX, A_MAX)
        x = torch.cat([(Th - T_TARGET) / 10, Ch / 50 - 1, a[:, None] / A_MAX], 1)
        Tn = Th[:, -1] + model(x)[:, 0] * 3
        c_new = (Ch[:, -1] + a).clamp(0, 100)
        Th = torch.cat([Th[:, 1:], Tn[:, None]], 1)
        Ch = torch.cat([Ch[:, 1:], c_new[:, None]], 1)
        out.append(Tn)
    return torch.stack(out, 1)


def plan_cost(Tpred, A):
    """행동열 하나의 계획 비용 = 예측 온도열의 스텝 비용 합 (N, H) → (N,)"""
    return step_cost(Tpred, A.clamp(-A_MAX, A_MAX)).sum(1)
