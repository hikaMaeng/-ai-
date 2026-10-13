"""6-2 노트북 04 · 05 공통 코드 — 움직이는 사각형 환경 · 작은 모델들 · 냉각수 밸브 시뮬레이터

노트북 셀 1 에서 `from wm_vision import *` 로 불러 쓴다. 외부 데이터 없음 : 환경은 전부 여기서 직접 만든다.
"""
import math
import random
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic") if f in _have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False})

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_GPU = DEV.type == "cuda"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if IS_GPU:
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


# ══════════════════════════════════════════════════════════════════════════
# 1. 움직이는 사각형 환경
#    32×32 RGB 프레임 · 4×4 색 사각형 · 한 번에 2픽셀 · 행동 5가지 · 벽에서는 못 나감(제자리)
#    배경에는 행동과 상관없이 프레임마다 새로 깜빡이는 약한 점 3개(방해 요소)
# ══════════════════════════════════════════════════════════════════════════
S, SQ, STEP = 32, 4, 2
MAXP = S - SQ                                  # 사각형 왼쪽 위 좌표의 최댓값(28)
GRID = MAXP // STEP + 1                        # 2픽셀 격자 위 시작 위치 15 × 15
ACTIONS = ["정지", "상", "하", "좌", "우"]
DELTA = np.array([[0, 0], [0, -STEP], [0, STEP], [-STEP, 0], [STEP, 0]])   # (dx, dy)
COLORS = np.array([[1, .2, .2], [.2, 1, .2], [.3, .3, 1], [1, 1, .2]], np.float32)
DOT_LEVEL, N_DOTS = 0.2, 3
HALF = (SQ - 1) / 2                            # 왼쪽 위 좌표 → 사각형 중심


def step_pos(pos, a):
    """진짜 환경 한 걸음. pos [N,2] (x,y), a [N] → 새 위치. 벽 밖으로 나가는 행동은 제자리(정보가 사라짐)"""
    new = pos + DELTA[a]
    blocked = ((new < 0) | (new > MAXP)).any(1)
    return np.where(blocked[:, None], pos, new)


def render(pos, col, rng, dots=N_DOTS):
    """위치 · 색 → 프레임 [N,3,S,S] (0~1). 깜빡이는 점은 rng 로 매번 새로 뽑는다"""
    n = len(pos)
    img = np.zeros((n, 3, S, S), np.float32)
    for _ in range(dots):
        dx, dy = rng.integers(0, S, n), rng.integers(0, S, n)
        img[np.arange(n), :, dy, dx] = DOT_LEVEL
    ar = np.arange(S)
    mx = (ar[None] >= pos[:, :1]) & (ar[None] < pos[:, :1] + SQ)
    my = (ar[None] >= pos[:, 1:]) & (ar[None] < pos[:, 1:] + SQ)
    m = my[:, :, None] & mx[:, None, :]
    return np.where(m[:, None], COLORS[col][:, :, None, None], img).astype(np.float32)


def to_u8(img):
    return (img * 255 + 0.5).astype(np.uint8)


def make_pairs(n, seed, dots=N_DOTS):
    """(프레임 t, 프레임 t+1) 쌍 n 개. a(진짜 행동)는 평가용으로만 따로 저장한다"""
    rng = np.random.default_rng(seed)
    pos0 = rng.integers(0, GRID, (n, 2)) * STEP
    col = rng.integers(0, len(COLORS), n)
    a = rng.integers(0, 5, n)
    pos1 = step_pos(pos0, a)
    f0 = np.empty((n, 3, S, S), np.uint8)
    f1 = np.empty_like(f0)
    for i in range(0, n, 10000):
        sl = slice(i, i + 10000)
        f0[sl] = to_u8(render(pos0[sl], col[sl], rng, dots))
        f1[sl] = to_u8(render(pos1[sl], col[sl], rng, dots))
    return dict(f0=f0, f1=f1, a=a, pos0=pos0, pos1=pos1, col=col)


def to_dev(d, keys):
    out = {}
    for k in keys:
        v = torch.from_numpy(np.ascontiguousarray(d[k]))
        out[k] = v.long().to(DEV) if k == "a" else v.to(DEV)
    return out


def fl(x):
    """uint8 프레임 → 0~1 실수"""
    return x.float() / 255


def show_frames(frames, titles=None, ncols=None, size=1.6, suptitle=None):
    """프레임 [N,3,S,S] (실수 0~1) 여러 장을 한 줄(또는 격자)로"""
    frames = [np.asarray(f) for f in frames]
    n = len(frames)
    ncols = ncols or n
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows * 1.15), squeeze=False)
    for i, ax in enumerate(axs.ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(np.clip(frames[i].transpose(1, 2, 0), 0, 1), interpolation="nearest")
            if titles:
                ax.set_title(titles[i], fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.show()


# ── 사각형 위치를 프레임에서 읽는 도구 : 가장 밝은 채널이 0.5 를 넘는 픽셀의 무게중심 ─────
def centroid(fr):
    """프레임 [N,3,S,S] → 사각형 중심 [N,2] (x,y). 사각형이 안 보이면 NaN(= 사각형을 잃어버림)"""
    w = (fr.max(1).values - 0.5).clamp(min=0)
    tot = w.sum((1, 2))
    xs = torch.arange(S, device=fr.device).float()
    cx = (w.sum(1) * xs).sum(1) / tot.clamp(min=1e-6)
    cy = (w.sum(2) * xs).sum(1) / tot.clamp(min=1e-6)
    c = torch.stack([cx, cy], 1)
    c[tot < 1.0] = float("nan")
    return c


def true_center(pos):
    return torch.tensor(pos, dtype=torch.float32, device=DEV) + HALF


def true_centers_after(pos0, acts):
    """진짜 환경으로 행동열 acts [N,H] 를 적용한 중심 위치 [N,H,2]"""
    pos, out = pos0.copy(), []
    for h in range(acts.shape[1]):
        pos = step_pos(pos, acts[:, h])
        out.append(pos.copy())
    return torch.tensor(np.stack(out, 1), dtype=torch.float32, device=DEV) + HALF


# ══════════════════════════════════════════════════════════════════════════
# 2. 동역학 모델(다음 프레임 예측) — 작은 CNN 하나, 행동을 넣는 방식만 바꾼다
#    cond = "none"   : 행동을 안 넣음
#           "concat" : 행동 임베딩을 입력 채널에 이어 붙임
#           "film"   : 층마다 정규화 뒤 (1+γ)·h + β 로 행동이 값을 조절(FiLM / AdaLN 계열)
# ══════════════════════════════════════════════════════════════════════════
class Dyn(nn.Module):
    def __init__(self, cond="none", n_act=5, D=16, ch=32, L=4):
        super().__init__()
        self.cond = cond
        self.emb = nn.Embedding(n_act, D) if (cond != "none" and n_act) else None
        cin = 3 + (D if cond == "concat" else 0)
        self.convs = nn.ModuleList([nn.Conv2d(cin if i == 0 else ch, ch, 3, padding=1) for i in range(L)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, ch) for _ in range(L)])
        self.film = nn.ModuleList([nn.Linear(D, 2 * ch) for _ in range(L)]) if cond == "film" else None
        if self.film is not None:
            for m in self.film:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, c=None):
        if self.cond != "none" and c.dtype == torch.long:
            c = self.emb(c)                                   # 행동 번호 → 벡터
        h = torch.cat([x, c[:, :, None, None].expand(-1, -1, S, S)], 1) if self.cond == "concat" else x
        for i, (cv, nm) in enumerate(zip(self.convs, self.norms)):
            h = nm(cv(h))
            if self.cond == "film":
                g, b = self.film[i](c).chunk(2, 1)
                h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
            h = F.silu(h)
        return x + self.out(h)                                # 현재 프레임 + 변화량


def train_dyn(model, D, steps, bs=128, lr=2e-3, seed=0, noise=0.0):
    """D : f0 · f1 · a (장치 위 텐서). noise > 0 이면 입력 프레임에만 가우시안 잡음을 더해 학습(정답은 깨끗한 프레임)"""
    set_seed(seed)
    model.to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * i / steps)) * 0.95 + 0.05)
    n = len(D["f0"])
    for i in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        x, y, a = fl(D["f0"][idx]), fl(D["f1"][idx]), D["a"][idx]
        if noise:
            x = x + noise * torch.randn_like(x)
        loss = F.mse_loss(model(x, a), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    return model.eval()


@torch.no_grad()
def mse_one_step(model, D, bs=1000):
    """학습에 안 쓴 쌍에서 진짜 행동을 넣었을 때 다음 프레임 예측의 픽셀당 MSE"""
    tot = 0.0
    for i in range(0, len(D["f0"]), bs):
        p = model(fl(D["f0"][i:i + bs]), D["a"][i:i + bs]).clamp(0, 1)
        tot += float(((p - fl(D["f1"][i:i + bs])) ** 2).mean()) * len(p)
    return tot / len(D["f0"])


@torch.no_grad()
def centers_all_actions(model, f0, bs=1000):
    """같은 프레임에 행동 5개를 각각 넣었을 때 예측 프레임의 사각형 중심 [N,5,2]"""
    out = []
    for i in range(0, len(f0), bs):
        x = fl(f0[i:i + bs])
        cs = [centroid(model(x, torch.full((len(x),), a, device=DEV)).clamp(0, 1)) for a in range(5)]
        out.append(torch.stack(cs, 1))
    return torch.cat(out)


def true_centers_all_actions(pos0):
    return torch.stack([true_center(step_pos(pos0, np.full(len(pos0), a))) for a in range(5)], 1)


def pair_sensitivity(cp, ct, tol=1.0):
    """행동 민감도 : 같은 프레임에서 행동 a, b 만 바꿨을 때, 진짜 결과가 다른 쌍(a≠b)에서
    (예측 중심 차이) 가 (진짜 중심 차이) 와 tol 픽셀 이내로 맞는 비율. 사각형을 잃으면(NaN) 불일치로 센다"""
    ok, tot = 0, 0
    for a in range(5):
        for b in range(a + 1, 5):
            dt = ct[:, a] - ct[:, b]
            differ = dt.abs().sum(1) > 0
            dp = cp[:, a] - cp[:, b]
            match = ((dp - dt).norm(dim=1) <= tol) & ~torch.isnan(dp).any(1)
            ok += int((match & differ).sum())
            tot += int(differ.sum())
    return ok / max(tot, 1)


def center_match(cp, ct, tol=1.0):
    """예측 중심이 진짜 다음 중심의 tol 픽셀 이내인 비율(행동별 벡터). cp, ct [N,A,2] → [A]"""
    ok = ((cp - ct).norm(dim=2) <= tol) & ~torch.isnan(cp).any(2)
    return ok.float().mean(0).cpu().numpy()


@torch.no_grad()
def rollout_centers(model, f0, acts, bs=500):
    """개루프 롤아웃 : 처음 한 장만 진짜 프레임, 이후에는 모델이 낸 프레임을 다시 입력으로.
    acts [N,H] (행동 번호, 장치 위 long) → 매 스텝 예측 사각형 중심 [N,H,2]"""
    res = []
    for i in range(0, len(f0), bs):
        x = fl(f0[i:i + bs])
        cs = []
        for h in range(acts.shape[1]):
            x = model(x, acts[i:i + bs, h]).clamp(0, 1)
            cs.append(centroid(x))
        res.append(torch.stack(cs, 1))
    return torch.cat(res)


def pos_error(cp, ct):
    """위치 오차(픽셀) : 사각형을 잃은 항목은 빼고 평균, 잃은 비율을 함께 돌려준다 → (평균 오차 [H], 잃은 비율 [H])"""
    lost = torch.isnan(cp).any(2)
    err = (cp - ct).norm(dim=2)
    err[lost] = float("nan")
    return torch.nanmean(err, 0).cpu().numpy(), lost.float().mean(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════
# 3. 벡터 양자화(VQ) · 잠재 행동 모델(LAM)
# ══════════════════════════════════════════════════════════════════════════
def vq_quantize(z, codebook, beta=0.25):
    """z [B,D] 를 코드북 [K,D] 에서 가장 가까운 행으로 치환.
    직통 경사(straight-through) : 앞으로는 q 를 쓰고, 뒤로는 z 로 경사가 그대로 흐른다.
    손실 = 코드북을 z 쪽으로 + β·z 를 코드북 쪽으로(커밋 항)"""
    idx = torch.cdist(z, codebook).argmin(1)
    q = codebook[idx]
    loss = F.mse_loss(q, z.detach()) + beta * F.mse_loss(z, q.detach())
    return z + (q - z).detach(), idx, loss


class LAM(nn.Module):
    """잠재 행동 모델 : 인코더(f_t, f_t+1 → 벡터) → VQ(코드 하나) → 디코더(=동역학: f_t + 코드 → f_t+1)"""

    def __init__(self, K, D=16, ch=32):
        super().__init__()
        self.K = K
        self.enc = nn.Sequential(
            nn.Conv2d(6, ch, 3, 2, 1), nn.SiLU(), nn.Conv2d(ch, ch, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(ch * 16, 64), nn.SiLU(), nn.Linear(64, D))
        self.codebook = nn.Embedding(K, D)
        self.dyn = Dyn("concat", n_act=0, D=D)

    def forward(self, x0, x1):
        z = self.enc(torch.cat([x0, x1], 1))
        q, idx, vql = vq_quantize(z, self.codebook.weight)
        return self.dyn(x0, q), idx, vql, z


def train_lam(K, f0, f1, steps, seed=0, bs=128, lr=1e-3, reset_until=0.6):
    """행동 라벨 없이 (f0, f1) 만으로 학습한다 — 인자에 진짜 행동이 없다.
    안 쓰이는 코드는 100 스텝마다(학습 앞 60% 동안만) 현재 인코더 출력으로 다시 심는다(죽은 코드 되살리기)"""
    set_seed(seed)
    m = LAM(K).to(DEV).train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * i / steps)) * 0.95 + 0.05)
    n, cnt, rec_log = len(f0), torch.zeros(K, device=DEV), []
    for i in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        x0, x1 = fl(f0[idx]), fl(f1[idx])
        z = m.enc(torch.cat([x0, x1], 1))
        if i == 0:
            m.codebook.weight.data = z.detach()[torch.randperm(bs, device=DEV)[:K]].clone()
        q, code, vql = vq_quantize(z, m.codebook.weight)
        rec = ((m.dyn(x0, q) - x1) ** 2).flatten(1).sum(1).mean()
        loss = rec + vql
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        cnt += torch.bincount(code, minlength=K).float()
        rec_log.append(float(rec))
        if i % 100 == 99:
            if i < reset_until * steps:
                dead = cnt == 0
                if dead.any():
                    pick = z.detach()[torch.randint(0, bs, (int(dead.sum()),), device=DEV)]
                    m.codebook.weight.data[dead] = pick + 0.01 * torch.randn_like(pick)
            cnt.zero_()
    m.eval()
    m.final_rec = float(np.mean(rec_log[-100:]))
    return m


@torch.no_grad()
def lam_codes(m, f0, f1, bs=1000):
    out = []
    for i in range(0, len(f0), bs):
        _, idx, _, _ = m(fl(f0[i:i + bs]), fl(f1[i:i + bs]))
        out.append(idx)
    return torch.cat(out).cpu().numpy()


def code_stats(code, label, K, n_label=5):
    """코드 ↔ 숨은 라벨의 일치도. C[라벨, 코드] 혼동행렬 · 순도 · 상호정보량(비트) · 안 쓰이는 코드 수"""
    C = np.zeros((n_label, K), int)
    np.add.at(C, (label, code), 1)
    N = C.sum()
    p = C / N
    pa, pc = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log2(p[nz] / (pa @ pc)[nz])).sum())
    Ha = float(-(pa[pa > 0] * np.log2(pa[pa > 0])).sum())
    Hc = float(-(pc[pc > 0] * np.log2(pc[pc > 0])).sum())
    return dict(C=C, per_code_purity=C.max(0) / np.maximum(C.sum(0), 1), purity=float(C.max(0).sum() / N), mi=mi, H_label=Ha,
                mi_ratio=mi / Ha, dead=int((C.sum(0) == 0).sum()), perplexity=2 ** Hc)


@torch.no_grad()
def lam_rollout(m, f0, codes, bs=1):
    """코드열(리스트)을 차례로 넣어 프레임을 이어 만든다 → 프레임 리스트(처음 프레임 포함)"""
    x = fl(f0)
    frames = [x[0].cpu().numpy()]
    for c in codes:
        x = m.dyn(x, m.codebook.weight[c][None]).clamp(0, 1)
        frames.append(x[0].cpu().numpy())
    return frames


# ══════════════════════════════════════════════════════════════════════════
# 4. 반복 마스크 채우기(MaskGIT 축소판) : 작은 VQ 토크나이저 + 마스크 트랜스포머
# ══════════════════════════════════════════════════════════════════════════
class TokVQ(nn.Module):
    """프레임 32×32 → 4×4 토큰 격자(토큰 하나 = 8×8 조각, 코드북 K개) → 다시 프레임"""

    def __init__(self, K=128, D=32, ch=64):
        super().__init__()
        self.K = K
        self.enc = nn.Sequential(nn.Conv2d(3, ch // 2, 4, 2, 1), nn.SiLU(), nn.Conv2d(ch // 2, ch, 4, 2, 1), nn.SiLU(),
                                 nn.Conv2d(ch, ch, 4, 2, 1), nn.SiLU(), nn.Conv2d(ch, D, 1))
        self.codebook = nn.Embedding(K, D)
        self.dec = nn.Sequential(nn.ConvTranspose2d(D, ch, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1), nn.SiLU(),
                                 nn.ConvTranspose2d(ch // 2, 3, 4, 2, 1))

    def forward(self, x):
        z = self.enc(x)
        B, D, h, w = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, D)
        q, idx, vql = vq_quantize(zf, self.codebook.weight)
        return self.dec(q.reshape(B, h, w, D).permute(0, 3, 1, 2)), idx.reshape(B, h * w), vql, zf

    @torch.no_grad()
    def encode(self, x):
        return self.forward(x)[1]

    @torch.no_grad()
    def decode(self, tok):
        B = tok.shape[0]
        q = self.codebook(tok).reshape(B, 4, 4, -1).permute(0, 3, 1, 2)
        return self.dec(q)


def train_tok(frames_u8, steps, seed=0, bs=128, lr=1e-3, reset_until=0.7):
    set_seed(seed)
    m = TokVQ().to(DEV).train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * i / steps)) * 0.95 + 0.05)
    n, cnt = len(frames_u8), torch.zeros(m.K, device=DEV)
    for i in range(steps):
        x = fl(frames_u8[torch.randint(0, n, (bs,), device=DEV)])
        rec_img, tok, vql, zf = m(x)
        if i == 0:
            m.codebook.weight.data = zf.detach()[torch.randperm(len(zf), device=DEV)[:m.K]].clone()
        loss = ((rec_img - x) ** 2).flatten(1).sum(1).mean() + vql
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        cnt += torch.bincount(tok.flatten(), minlength=m.K).float()
        if i % 100 == 99:
            if i < reset_until * steps:
                dead = cnt == 0
                if dead.any():
                    pick = zf.detach()[torch.randint(0, len(zf), (int(dead.sum()),), device=DEV)]
                    m.codebook.weight.data[dead] = pick + 0.01 * torch.randn_like(pick)
            cnt.zero_()
    return m.eval()


class MaskTf(nn.Module):
    """양방향 트랜스포머 : [행동 토큰] + 이전 프레임 토큰 16개(보임) + 다음 프레임 토큰 16개(일부는 [MASK]) → 가려진 칸의 토큰"""

    def __init__(self, K, T=16, d=128, L=4, heads=4, use_act=True):
        super().__init__()
        self.K, self.T, self.use_act = K, T, use_act
        self.tok = nn.Embedding(K + 1, d)                        # 마지막 번호 K = [MASK]
        self.pos = nn.Parameter(torch.randn(2 * T, d) * 0.02)
        self.act = nn.Embedding(5, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.net = nn.TransformerEncoder(layer, L)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, K))

    def forward(self, prev, tgt, a):
        h = self.tok(torch.cat([prev, tgt], 1)) + self.pos
        if self.use_act:
            h = torch.cat([self.act(a)[:, None], h], 1)
        h = self.net(h)
        return self.head(h[:, -self.T:])


def train_mask_tf(tf, prev_tok, tgt_tok, act, steps, seed=0, bs=256, lr=1e-3):
    set_seed(seed)
    tf.to(DEV).train()
    opt = torch.optim.AdamW(tf.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * i / steps)) * 0.95 + 0.05)
    n, T = len(prev_tok), tf.T
    for i in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        prev, tgt, a = prev_tok[idx], tgt_tok[idx], act[idx]
        ratio = torch.cos(math.pi / 2 * torch.rand(bs, device=DEV))                 # 가릴 비율(코사인 일정)
        n_mask = (ratio * T).ceil().clamp(min=1).long()
        rank = torch.rand(bs, T, device=DEV).argsort(1).argsort(1)
        masked = rank < n_mask[:, None]
        logits = tf(prev, tgt.masked_fill(masked, tf.K), a)
        loss = F.cross_entropy(logits[masked], tgt[masked])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    return tf.eval()


@torch.no_grad()
def mask_decode(tf, prev, a, R):
    """전부 [MASK] 에서 시작해 R 번에 나눠 채운다. 매 번 모든 칸의 토큰을 예측하고, 가려진 칸 중 확신(최대 확률)이 높은 칸부터 확정.
    R=1 은 한 번에 전부 예측. 남길 가림 칸 수는 코사인 일정으로 줄인다"""
    B, T = prev.shape
    tgt = torch.full((B, T), tf.K, device=prev.device, dtype=torch.long)
    for r in range(1, R + 1):
        p = tf(prev, tgt, a).softmax(-1)
        conf, tok = p.max(-1)
        masked = tgt == tf.K
        n_now = int(T * math.cos(math.pi / 2 * (r - 1) / R))
        n_next = int(T * math.cos(math.pi / 2 * r / R))
        top = conf.masked_fill(~masked, -1).topk(n_now - n_next, dim=1).indices
        tgt = tgt.scatter(1, top, tok.gather(1, top))
    return tgt


# ══════════════════════════════════════════════════════════════════════════
# 5. 냉각수 밸브 시뮬레이터(05 현장 성과용)
#    상태 : 온도 T · 개도 u(0~100%) · 열부하 L(밝혀진 센서값, 가끔 바뀜)
#    행동 : 개도 변화 Δu ∈ [-10, +10] — 효과는 2스텝 뒤에 개도에 반영(대기열)
#    T 는 평형온도 T_eq(u, L) = 60 + 40·L − 0.4·u 쪽으로 20% 씩 다가간다. 측정 잡음 0.3℃
# ══════════════════════════════════════════════════════════════════════════
V_TARGET, V_LIMIT, V_OBS_NOISE = 70.0, 80.0, 0.3


def valve_teq(u, L):
    return 60 + 40 * L - 0.4 * u


def valve_step_true(T, u, q1, L):
    """잡음 없는 진짜 한 걸음(오라클 월드모델이 씀) → (T', u')"""
    u2 = np.clip(u + q1, 0, 100) if isinstance(u, np.ndarray) else torch.clamp(u + q1, 0, 100)
    return T + 0.2 * (valve_teq(u2, L) - T), u2


class Valve:
    def __init__(self, n, seed):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.L = self.rng.uniform(0.3, 0.9, n)
        self.T = self.rng.uniform(60, 85, n)
        self.u = self.rng.uniform(20, 80, n)
        self.q = np.zeros((n, 2))                             # 아직 반영 안 된 행동 2개(q[:,0] 이 이번에 반영)

    def obs(self):
        return dict(T=self.T + self.rng.normal(0, V_OBS_NOISE, self.n), u=self.u.copy(), q=self.q.copy(), L=self.L.copy())

    def step(self, a):
        a = np.clip(a, -10, 10)
        self.u = np.clip(self.u + self.q[:, 0], 0, 100)
        self.q = np.stack([self.q[:, 1], a], 1)
        self.T = self.T + 0.2 * (valve_teq(self.u, self.L) - self.T) + self.rng.normal(0, 0.1, self.n)
        jump = self.rng.random(self.n) < 0.04
        self.L = np.where(jump, self.rng.uniform(0.3, 0.9, self.n), self.L)


def valve_feats(T, u, q1, q2, L, a, delay=True):
    """MLP 입력. delay=False 면 대기열(q1, q2)을 못 보는 모델(지연을 모르는 모델)"""
    cols = [T / 100, u / 100, L] + ([q1 / 10, q2 / 10] if delay else []) + [a / 10]
    return torch.stack(cols, -1).float()


class ValveMLP(nn.Module):
    """(T, u, [q1, q2], L, a) → 다음 (T, u). 출력은 변화량"""

    def __init__(self, delay=True):
        super().__init__()
        self.delay = delay
        self.net = nn.Sequential(nn.Linear(6 if delay else 4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, T, u, q1, q2, L, a):
        d = self.net(valve_feats(T, u, q1, q2, L, a, self.delay))
        return T + 5 * d[..., 0], torch.clamp(u + 10 * d[..., 1], 0, 100)


def valve_dataset(n_ep, seed, steps=100):
    """무작위 행동으로 굴린 전이 자료 : 입력 (T_관측, u, q1, q2, L, a) → 다음 관측 (T', u')"""
    env, rng = Valve(n_ep, seed), np.random.default_rng(seed + 1)
    rows = []
    for _ in range(steps):
        o = env.obs()
        a = rng.uniform(-10, 10, n_ep)
        env.step(a)
        o2 = env.obs()
        rows.append(np.stack([o["T"], o["u"], o["q"][:, 0], o["q"][:, 1], o["L"], a, o2["T"], o2["u"]], 1))
    return torch.tensor(np.concatenate(rows), dtype=torch.float32)


def train_valve(data, delay=True, steps=3000, seed=0, bs=256, lr=3e-3):
    set_seed(seed)
    m = ValveMLP(delay)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        b = data[torch.randint(0, len(data), (bs,))]
        T2, u2 = m(*[b[:, i] for i in range(6)])
        loss = F.mse_loss((T2 - b[:, 0]) / 5, (b[:, 6] - b[:, 0]) / 5) + F.mse_loss((u2 - b[:, 1]) / 10, (b[:, 7] - b[:, 1]) / 10)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


def valve_rmse(m, data):
    with torch.no_grad():
        T2, u2 = m(*[data[:, i] for i in range(6)])
    return float(((T2 - data[:, 6]) ** 2).mean().sqrt()), float(((u2 - data[:, 7]) ** 2).mean().sqrt())


def oracle_model(T, u, q1, q2, L, a):
    return valve_step_true(T, u, q1, L)


@torch.no_grad()
def mpc_action(model, o, n_cand=256, H=8, gen=None):
    """무작위 표본 MPC : 후보 행동열 n_cand 개를 월드모델 안에서 H 스텝 굴려 (T−70)² 합이 가장 작은 열의 첫 행동을 쓴다"""
    E = len(o["T"])
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    T, u, L = t(o["T"])[:, None].expand(E, n_cand), t(o["u"])[:, None].expand(E, n_cand), t(o["L"])[:, None].expand(E, n_cand)
    q1, q2 = t(o["q"][:, 0])[:, None].expand(E, n_cand), t(o["q"][:, 1])[:, None].expand(E, n_cand)
    A = torch.rand(E, n_cand, H, generator=gen) * 20 - 10
    cost = torch.zeros(E, n_cand)
    for h in range(H):
        T, u = model(T, u, q1, q2, L, A[:, :, h])
        q1, q2 = q2, A[:, :, h]
        cost += (T - V_TARGET) ** 2
    best = cost.argmin(1)
    return A[torch.arange(E), best, 0].numpy()


def run_valve(policy, n_env=20, steps=100, seed=123):
    """정책 policy(obs) → 행동 [E] 로 steps 스텝. 진짜 온도 기준 (평균 비용 (T−70)², 상한(80℃) 넘은 스텝 수의 평균, 평균 절대 오차)"""
    env = Valve(n_env, seed)
    cost, viol, absd = np.zeros(n_env), np.zeros(n_env), np.zeros(n_env)
    for _ in range(steps):
        env.step(policy(env.obs()))
        cost += (env.T - V_TARGET) ** 2
        viol += env.T > V_LIMIT
        absd += np.abs(env.T - V_TARGET)
    return dict(cost=float((cost / steps).mean()), viol=float(viol.mean()), mae=float((absd / steps).mean()))


def ms(x, d=3):
    """평균±표준편차 문자열"""
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f}±{x.std():.{d}f}"


def coherent_rate(fr):
    """프레임 [N,3,S,S] 중 밝은 픽셀(>0.5)이 사각형 넓이(16)의 0.6~1.4배(10~22개)이고 중심이 잡히는 비율 = 사각형 하나가 온전한 정도"""
    n = (fr.max(1).values > 0.5).flatten(1).sum(1)
    return float(((n >= 10) & (n <= 22) & ~torch.isnan(centroid(fr)).any(1)).float().mean())
