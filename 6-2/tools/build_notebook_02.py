"""6-2 실습 노트북 02 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook_02.py   → ../notebooks/02_JEPA_가린_부분의_특징_예측.ipynb (출력 없이)
"""
from nb import Book

b = Book("02_JEPA_가린_부분의_특징_예측.ipynb")

b.md('''
# 실습 02 · JEPA — 가린 부분을 「그리지」 않고 「특징」으로 예측하기

**직접 해 보는 것** : 16×16 합성 이미지에서 조각(패치)을 가리고, 가려진 곳을 (a) **픽셀로 복원**하는 생성형 모델과 (b) **특징으로 예측**하는 JEPA 를 같은 구조·같은 예산으로 학습해 비교한다. 이미지에는 위치가 정답인 **물체**와, 아무리 봐도 맞힐 수 없는 **잡음 영역**이 함께 있다. 그리고 JEPA 가 학습 중에 **붕괴**(모든 입력에 같은 특징을 내는 것)하는 조건을 직접 만들어 본다.

**강의의 개념과 이어지는 곳** : 생성형(픽셀 복원) vs 표현예측형(JEPA) · 예측기와 마스크 토큰 · 컨텍스트 인코더와 타깃 인코더 · L1 손실 · 가려진 위치에만 거는 손실 · 경사 차단(stop-gradient) · EMA 타깃 · 표현 붕괴.

| 셀 | 하는 일 |
|---|---|
| 1 | 설정 |
| 2 | 합성 데이터와 마스크 그림 |
| 3 | L1 손실을 손으로 계산 = `F.l1_loss` |
| 4 | 생성형 기준선 — 픽셀 복원(MSE), 잡음 영역이 손실에서 차지하는 몫 |
| 5 | JEPA — 컨텍스트 인코더 · 예측기(마스크 토큰) · EMA 타깃 인코더 · 가려진 위치 특징의 L1 |
| 6 | 선형 프로브(탐침) — 인코더 특징으로 물체 위치를 맞히는 R², 생성형 vs JEPA |
| 7 | 붕괴 실험 — 경사 차단 제거 · EMA 0 |
| 8 | 정리 · 확인 질문 3개 |

- 실행 : **Run → Run All Cells**. GPU 없어도 된다(CPU 에서 더 오래 걸림). 외부 데이터 없이 직접 만든 합성 이미지만 쓴다
- JEPA 는 **화면을 그리지 않는다.** 그래서 학습이 잘 됐는지 눈으로 확인할 수 없고, 특징에 정보가 들어 있는지를 따로 측정하는 **프로브**가 필요하다 — 이 노트북의 중심 주제다
''')

b.md('''
### 셀 1 · 설정

**왜 이 셀을 도는가** : 결과를 바꾸는 값(시드 · 학습 스텝 · 학습률 · 크기)을 한곳에 모아 찍는다.

**약속** : 한 번의 학습은 **새로 만든 이미지 묶음**을 매 스텝 쓴다(합성 데이터라 무한히 뽑을 수 있어 외우기 걱정이 없다). 이미지의 절반인 8조각을 가리고 8조각만 보여 준다.
''')
b.code('''
# 셀 1 · 설정
import sys; sys.path.insert(0, ".")
from wm_common import *
import copy

SEEDS = (0, 1, 2)         # 학습 시드 3개
STEPS = 800               # 학습 스텝(한 번의 학습)
BATCH = 128               # 스텝마다 새로 만든 이미지 수
LR = 1e-3                 # AdamW 학습률
DIM = 64                  # 특징 차원
EMA_TAU = 0.99            # EMA 계수 : 타깃 = 0.99·타깃 + 0.01·컨텍스트
N_VISIBLE = 8             # 보이는 조각 수(전체 16조각 중)
LOG_EVERY = 100           # 곡선을 기록하는 간격
PROBE_TRAIN, PROBE_TEST = 4000, 1000     # 프로브용 이미지 수
RIDGE = 10.0              # 프로브 릿지 정규화 세기

print("장치 :", device_info())
print(f"시드 {SEEDS} · 스텝 {STEPS} · 배치 {BATCH} · 학습률 {LR} · 특징 차원 {DIM} · EMA {EMA_TAU} · 보이는 조각 {N_VISIBLE}/16 · 프로브 학습 {PROBE_TRAIN}장 / 시험 {PROBE_TEST}장 · 릿지 {RIDGE}")
''')

b.md('''
### 셀 2 · 합성 데이터와 마스크

**왜 이 셀을 도는가** : 실험의 무대. 이미지 한 장은 이렇게 만들어진다.
- **물체** : 밝은 점(핵) 하나와 그 둘레의 넓고 흐린 **후광**. 위치 (x, y)가 매번 다르다 — 이 위치가 이 이미지가 가진 「의미 있는 정보」(정답). 밝기와 후광 크기도 이미지마다 달라서(위치와 무관) 위치를 픽셀 값의 단순한 가중합으로는 읽어 낼 수 없다
- **잡음 영역** : 이미지 맨 위 띠(4줄 = 조각 4개)는 매 이미지마다 **서로 무관한 무작위 질감**으로 채운다. 밝기도 물체보다 눈에 띄지만, 아무리 주변을 봐도 **맞힐 수 없다**
- 조각 = 4×4 픽셀 → 한 이미지 16조각. 조각 번호 0~3 이 잡음 띠, 4~15 가 물체가 놓이는 영역

**읽는 법** : 아래 그림의 윗줄은 원본, 아랫줄은 모델에게 보여 주는 모습(검게 가린 곳이 가려진 조각, 빨간 선 = 잡음 띠). 물체의 핵이 가려져도 보이는 조각의 후광에서 위치를 추정할 수 있다는 점이 핵심 — 후광이 「주변에서 맞힐 수 있는 정보」의 역할.
''')
b.code('''
# 셀 2 · 합성 데이터와 마스크
G = torch.arange(16, device=DEVICE).float()
YY, XX = torch.meshgrid(G, G, indexing="ij")

def make_images(n, gen=None):
    """이미지 n장과 물체 위치(x, y). 물체 = 밝기 × [핵(σ 0.8) + 후광(σ 2.5~4.5, 진폭 0.5)], 맨 위 4줄 = 무작위 잡음"""
    r = lambda *s: torch.rand(*s, generator=gen, device=DEVICE)
    cx, cy = r(n) * 12 + 2, r(n) * 9 + 5              # x 2~14, y 5~14 (잡음 띠를 피함)
    amp, sig = (r(n) * 0.8 + 0.6)[:, None, None], (r(n) * 2 + 2.5)[:, None, None]       # 밝기 0.6~1.4 · 후광 크기 σ 2.5~4.5 : 위치와 무관하게 이미지마다 다름
    d2 = (XX[None] - cx[:, None, None]) ** 2 + (YY[None] - cy[:, None, None]) ** 2
    img = amp * (torch.exp(-d2 / (2 * 0.8 ** 2)) + 0.5 * torch.exp(-d2 / (2 * sig ** 2)))
    img[:, :4, :] = r(n, 4, 16)                        # 잡음 띠 : 이미지마다 독립, 0~1 균등
    return img, torch.stack([cx, cy], 1)

def patchify(img):
    """(n,16,16) → (n,16,16) : 조각 16개 × 조각당 픽셀 16개. 조각 번호 = 행 블록 × 4 + 열 블록"""
    n = img.shape[0]
    return img.reshape(n, 4, 4, 4, 4).permute(0, 1, 3, 2, 4).reshape(n, 16, 16)

def unpatchify(p):
    n = p.shape[0]
    return p.reshape(n, 4, 4, 4, 4).permute(0, 1, 3, 2, 4).reshape(n, 16, 16)

def random_split(n, gen=None):
    """조각을 무작위로 섞어 앞 N_VISIBLE 개 = 보임, 나머지 = 가림 (n, 8) 두 묶음"""
    order = torch.rand(n, 16, generator=gen, device=DEVICE).argsort(1)
    return order[:, :N_VISIBLE], order[:, N_VISIBLE:]

g = torch.Generator(device=DEVICE); g.manual_seed(123)
img, pos = make_images(6, g); vis, msk = random_split(6, g)
fig, ax = plt.subplots(2, 6, figsize=(12, 4.4))
for i in range(6):
    ax[0, i].imshow(img[i].cpu(), cmap="gray", vmin=0, vmax=1.2); ax[0, i].set_title(f"원본 · 물체 ({pos[i, 0]:.1f}, {pos[i, 1]:.1f})", fontsize=9)
    shown = patchify(img[i:i + 1]).clone(); shown[0, msk[i]] = -0.3                     # 가린 조각은 검게
    ax[1, i].imshow(unpatchify(shown)[0].cpu(), cmap="gray", vmin=-0.3, vmax=1.2); ax[1, i].set_title("모델이 보는 것(검정 = 가림)", fontsize=9)
    for a in ax[:, i]:
        a.axhline(3.5, color="red", lw=1); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
print("조각 0~3 = 맨 위 잡음 띠(빨간 선 위) · 한 이미지의 가린 조각 번호 예 :", sorted(msk[0].tolist()))
''')

b.md('''
### 셀 3 · L1 손실을 손으로 계산

**왜 이 셀을 도는가** : JEPA 의 손실은 **예측한 특징과 정답 특징의 차이의 절댓값**(L1)이다. 가장 작은 예(숫자 2개)로 손 계산과 `F.l1_loss` 가 같은지 확인한다.

**읽는 법** : 예측 [0.2, 0.8], 정답 [0.5, 0.6] 이면 차이 절댓값이 0.3 과 0.2. **합**은 0.5, **평균**은 0.25. `F.l1_loss` 의 기본값은 평균이다(이 노트북의 학습도 평균). 합으로 쓰는 `reduction="sum"` 을 주면 0.5 가 나온다.
''')
b.code('''
# 셀 3 · L1 손 계산 = torch
pred, tgt = torch.tensor([0.2, 0.8]), torch.tensor([0.5, 0.6])
hand_sum = abs(0.2 - 0.5) + abs(0.8 - 0.6)
print(f"손 계산 : |0.2-0.5| + |0.8-0.6| = 0.3 + 0.2 = {hand_sum:.2f}   (평균 = {hand_sum / 2:.2f})")
print(f"F.l1_loss(기본 = 평균)       = {F.l1_loss(pred, tgt).item():.2f}")
print(f"F.l1_loss(reduction='sum')   = {F.l1_loss(pred, tgt, reduction='sum').item():.2f}")
''')

b.md('''
### 셀 4 · 생성형 기준선 — 가려진 조각의 픽셀을 복원

**왜 이 셀을 도는가** : 가린 곳을 **화면 그대로 다시 그리는** 방식(생성형)을 먼저 만든다. 구조는 JEPA 와 같다. 보이는 조각만 인코더에 넣고, 가려진 위치마다 **마스크 토큰**(「여기를 채워라」라는 학습되는 빈 칸 표시)을 붙여 예측기에 넣고, 출력으로 **픽셀 16개**를 내서 정답 픽셀과의 **MSE**(제곱 오차)를 줄인다. 손실은 **가려진 조각에만** 건다.

**왜 낭비가 생기는가** : 잡음 띠는 이미지마다 무작위라서 가려졌을 때 맞힐 방법이 없다. 그런데 픽셀 손실에서는 이 조각도 똑같이 센다 — 최선이어도 평균값(0.5)으로 찍어 오차 약 0.083(균등분포의 분산)이 남고, 모델은 이 못 줄이는 오차를 줄이려 용량을 쓴다. 측정할 것: 학습 후 **가려진 조각의 픽셀 오차 중 잡음 띠가 차지하는 몫**.

**읽는 법** : 아래 두 번째 그림 줄에서 잡음 띠 복원은 회색 뭉개짐(평균값)이다. 표의 몫이 잡음 띠가 차지하는 면적 비율(가려진 조각 중 약 25%)보다 훨씬 크면 손실의 대부분이 맞힐 수 없는 곳에 쓰이고 있다는 뜻.
''')
b.code('''
# 셀 4 · 생성형 기준선과 모델 정의(JEPA 도 같은 부품을 씀)
class Encoder(nn.Module):
    """조각 → 특징. 조각 임베딩 + 위치 임베딩 + 작은 트랜스포머(2층). idx 를 주면 그 위치의 조각만 통과"""
    def __init__(s, dim=DIM, layers=2):
        super().__init__()
        s.emb = nn.Linear(16, dim); s.pos = nn.Parameter(torch.randn(16, dim) * 0.02)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 4, 2 * dim, 0.0, batch_first=True, norm_first=True), layers)
    def forward(s, patches, idx=None):
        x = s.emb(patches) + s.pos
        if idx is not None:
            x = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
        return s.tf(x)

class Predictor(nn.Module):
    """보이는 조각의 특징 + 가려진 위치의 마스크 토큰 → 가려진 위치의 출력. out = 16 이면 픽셀(생성형), out = DIM 이면 특징(JEPA)"""
    def __init__(s, out, dim=DIM, layers=2):
        super().__init__()
        s.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02); s.pos = nn.Parameter(torch.randn(16, dim) * 0.02)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 4, 2 * dim, 0.0, batch_first=True, norm_first=True), layers)
        s.head = nn.Linear(dim, out)
    def forward(s, ctx, msk_idx):
        n, m = msk_idx.shape
        q = s.mask_token.expand(n, m, -1) + s.pos[msk_idx]                  # 마스크 토큰 + 「어디인가」를 알려 주는 위치 임베딩
        return s.head(s.tf(torch.cat([ctx, q], 1))[:, -m:])

# 평가용 고정 이미지(학습에 안 씀)
ge = torch.Generator(device=DEVICE); ge.manual_seed(2024)
EV_IMG, EV_POS = make_images(1024, ge); EV_VIS, EV_MSK = random_split(1024, ge); EV_P = patchify(EV_IMG)
take = lambda x, idx: torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))

def train_gen(seed, steps=STEPS):
    """생성형 : 보이는 조각 → 가려진 조각의 픽셀(MSE)"""
    seed_all(seed); gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
    enc, pred = Encoder().to(DEVICE), Predictor(16).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), LR); log = []
    for st in range(1, steps + 1):
        img, _ = make_images(BATCH, gen); vis, msk = random_split(BATCH, gen); P = patchify(img)
        out = pred(enc(P, vis), msk)
        loss = F.mse_loss(out, take(P, msk))                                # 가려진 조각의 픽셀만
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0); opt.step()
        if st % LOG_EVERY == 0:
            log.append(dict(step=st, loss=loss.item()))
    return enc.eval(), pred.eval(), log

t0 = time.time()
GEN = {0: train_gen(0)}
enc_g, pred_g, log_g = GEN[0]
print(f"생성형 학습(시드 0) {time.time() - t0:.0f}초 · 마지막 학습 손실(MSE) {log_g[-1]['loss']:.4f}")

@torch.no_grad()
def gen_error_share(enc, pred):
    """가려진 조각의 픽셀 MSE 를 잡음 띠(조각 0~3)와 물체 영역(4~15)으로 나눔"""
    se = ((pred(enc(EV_P, EV_VIS), EV_MSK) - take(EV_P, EV_MSK)) ** 2).mean(-1)            # (n, 8) 조각별 MSE
    noise = (EV_MSK < 4).float()
    tot = se.sum(); n_noise = (se * noise).sum()
    return dict(잡음띠_몫=float(n_noise / tot), 잡음띠_조각MSE=float((se * noise).sum() / noise.sum()), 물체영역_조각MSE=float((se * (1 - noise)).sum() / (1 - noise).sum()),
                가린조각중_잡음띠_비율=float(noise.mean()), 전체_가린조각_MSE=float(se.mean()))
sh_g = gen_error_share(enc_g, pred_g)
print(f"가려진 조각 중 잡음 띠 비율 {sh_g['가린조각중_잡음띠_비율']:.3f} (면적 몫)  vs  픽셀 오차 중 잡음 띠 몫 {sh_g['잡음띠_몫']:.3f}")
print(f"조각당 MSE : 잡음 띠 {sh_g['잡음띠_조각MSE']:.4f} (이론 바닥 = 균등분포 분산 1/12 = {1 / 12:.4f}) · 물체 영역 {sh_g['물체영역_조각MSE']:.4f}")

with torch.no_grad():
    sel = slice(0, 6); rec = EV_P[sel].clone(); rec[torch.arange(6)[:, None], EV_MSK[sel]] = pred_g(enc_g(EV_P[sel], EV_VIS[sel]), EV_MSK[sel])
    shown = EV_P[sel].clone(); shown[torch.arange(6)[:, None], EV_MSK[sel]] = -0.3
fig, ax = plt.subplots(3, 6, figsize=(12, 6.4))
for i in range(6):
    for r, (im, ttl) in enumerate(((EV_IMG[i], "원본"), (unpatchify(shown)[i], "보이는 것"), (unpatchify(rec)[i], "생성형 복원"))):
        ax[r, i].imshow(im.cpu(), cmap="gray", vmin=-0.3 if r == 1 else 0, vmax=1.2); ax[r, i].set_title(ttl, fontsize=9); ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
plt.tight_layout(); plt.show()
''')

b.md('''
### 셀 5 · JEPA — 가려진 곳의 특징을 예측

**왜 이 셀을 도는가** : 같은 구조에서 **출력이 픽셀이 아니라 특징**이 되는 것이 JEPA 다. 부품 네 개.
1. **컨텍스트 인코더** : 보이는 조각만 받아 특징을 만든다(학습됨)
2. **예측기** : 보이는 조각의 특징 + 가려진 위치의 **마스크 토큰**을 받아 **가려진 위치의 특징**을 낸다
3. **타깃 인코더** : **전체 이미지**를 받아 정답 특징을 만든다. 학습되지 않고 컨텍스트 인코더의 **EMA(지수이동평균)** — 매 스텝 `타깃 = 0.99·타깃 + 0.01·컨텍스트` 로만 따라간다
4. **손실** : 가려진 위치에서 (예측 특징 − 정답 특징)의 **L1**. 정답 쪽에는 **경사 차단**(`torch.no_grad`) — 정답을 예측 쪽으로 끌어오는 길을 막는다. 정답 특징에는 층 정규화(평균 0 · 분산 1)를 씌워 크기로 손실을 줄이는 지름길을 막는다

**읽는 법** : 학습은 곡선으로만 확인된다. 이 셀은 변종(경사 차단 · EMA · 타깃 정규화)을 인자로 받는 한 함수로 만들어, 셀 7 의 붕괴 실험에서 그대로 다시 쓴다. 곡선 세 개 — **손실**, **특징 표준편차**(같은 위치의 특징이 이미지마다 얼마나 달라지는가 — 0 이면 붕괴), **프로브 R²**(셀 6 에서 설명) — 를 100스텝마다 기록한다.
''')
b.code('''
# 셀 5 · JEPA (붕괴 실험용 인자 포함)
@torch.no_grad()
def feature_std(enc, P):
    """같은 조각 위치의 특징이 이미지마다 얼마나 변하나 = (이미지 방향 표준편차)를 위치 · 차원에 걸쳐 평균. 0 에 가까우면 붕괴"""
    return enc(P).std(0).mean().item()

@torch.no_grad()
def eff_rank(enc, P):
    """유효 차원 수 : 같은 조각 위치의 특징(이미지 방향)의 공분산 고윳값 분포의 엔트로피 지수. 크기와 무관 — 특징이 몇 개 축으로 쪼그라들었나(1 에 가까우면 한 축, 최대 DIM)"""
    f = enc(P); f = f - f.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(torch.einsum("npd,npe->pde", f, f) / (len(P) - 1)).clamp_min(1e-20)         # (16, DIM)
    pr = ev / ev.sum(1, keepdim=True)
    return torch.exp(-(pr * pr.log()).sum(1)).mean().item()

@torch.no_grad()
def probe_r2(enc, n_tr=PROBE_TRAIN, n_te=PROBE_TEST, seed=777, feat=None):
    """선형 프로브 : 인코더(전체 이미지 입력) 특징을 16조각에 걸쳐 평균 내어 이미지당 DIM 개 숫자로 만들고 → 릿지 회귀로 물체 위치(x, y). 시험 이미지의 R² 평균"""
    gp = torch.Generator(device=DEVICE); gp.manual_seed(seed)
    img, pos = make_images(n_tr + n_te, gp)
    X = feat(img) if feat is not None else enc(patchify(img)).mean(1)
    mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0)
    X = (X - mu) / sd.clamp_min(1e-6) * (sd > 1e-6)                     # 특징을 표준화(크기가 작아져도 정보가 있으면 읽힘)
    Xtr, Xte = X[:n_tr], X[n_tr:]; ym = pos[:n_tr].mean(0)
    Xtr1, Xte1 = torch.cat([Xtr, torch.ones(n_tr, 1, device=DEVICE)], 1), torch.cat([Xte, torch.ones(n_te, 1, device=DEVICE)], 1)
    I = torch.eye(Xtr1.shape[1], device=DEVICE) * RIDGE; I[-1, -1] = 0
    W = torch.linalg.solve(Xtr1.T @ Xtr1 + I, Xtr1.T @ (pos[:n_tr] - ym))
    res = ((Xte1 @ W + ym) - pos[n_tr:]) ** 2
    r2 = 1 - res.sum(0) / ((pos[n_tr:] - pos[n_tr:].mean(0)) ** 2).sum(0)
    return float(r2.mean())

def train_jepa(seed, steps=STEPS, ema=EMA_TAU, stop_grad=True, target_norm=True, log_every=LOG_EVERY):
    """JEPA. stop_grad=False 이면 정답도 같은 인코더를 거쳐 학습에 끌려 옴(타깃 인코더 = 컨텍스트 인코더 그 자체).
    ema = 타깃 인코더가 컨텍스트 인코더를 따라가는 비율(0 이면 매 스텝 그대로 복사). target_norm = 정답 특징에 층 정규화"""
    seed_all(seed); gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
    enc, pred = Encoder().to(DEVICE), Predictor(DIM).to(DEVICE)
    tgt = copy.deepcopy(enc) if stop_grad else enc
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, LR); log, run = [], []
    for st in range(1, steps + 1):
        img, _ = make_images(BATCH, gen); vis, msk = random_split(BATCH, gen); P = patchify(img)
        p = pred(enc(P, vis), msk)                                                  # 예측 특징 (n, 8, DIM)
        if stop_grad:
            with torch.no_grad():                                                   # 경사 차단 : 정답 쪽으로는 기울기가 흐르지 않는다
                y = take(tgt(P), msk)
        else:
            y = take(tgt(P), msk)                                                   # 경사 차단 없음 : 정답도 학습으로 끌려 온다
        if target_norm:
            y = F.layer_norm(y, (DIM,))
        loss = F.l1_loss(p, y)                                                      # 가려진 위치만, L1(평균)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if stop_grad:
            with torch.no_grad():
                for a, c in zip(tgt.parameters(), enc.parameters()):
                    a.mul_(ema).add_(c, alpha=1 - ema)                              # EMA : ema=0 이면 a ← c(복사)
        run.append(loss.item())
        if st % log_every == 0:
            enc.eval()
            log.append(dict(step=st, loss=float(np.mean(run)), feat_std=feature_std(enc, EV_P), eff_rank=eff_rank(enc, EV_P), probe_r2=probe_r2(enc))); run = []
            enc.train()
    return enc.eval(), pred.eval(), tgt.eval(), log

t0 = time.time()
JEPA = {0: train_jepa(0)}
enc_j, pred_j, tgt_j, log_j = JEPA[0]
print(f"JEPA 학습(시드 0) {time.time() - t0:.0f}초")
print(pd.DataFrame(log_j).iloc[[0, 1, 3, 5, -1]].to_string(index=False))
''')

b.md('''
### 셀 6 · 선형 프로브 — 학습된 인코더가 위치를 알고 있는가

**왜 이 셀을 도는가** : JEPA 는 화면을 그리지 않으므로 **눈으로 확인할 방법이 없다**. 손실이 내려갔다는 것은 「예측이 정답에 가까워졌다」일 뿐, 특징에 쓸모 있는 정보가 있다는 보장이 아니다(셀 7 의 붕괴가 바로 손실만 0 이 되는 경우). 그래서 **인코더를 얼려 둔 채 특징 위에 선형 회귀 하나**를 얹어(이미지 한 장의 특징 16조각을 평균 내어 64개 숫자로 만든 뒤 그 위에), 물체 위치 (x, y)를 맞히는 정도(**R²**: 1 = 완벽, 0 = 평균값 찍기와 같음)를 측정한다. 이것이 프로브(탐침)다. 선형이기 때문에 R² 가 높다면 정보가 특징에 이미 풀려서 들어 있다는 뜻.

**비교** : 같은 구조·같은 스텝 수·같은 시드 3개의 **생성형 인코더** vs **JEPA 인코더**. 기준 둘을 곁들인다 — **학습 전 인코더**(무작위 초기값)와 **원본 픽셀**(256개 값에 직접 선형 회귀).

**읽는 법** : 어느 쪽이 높든 결과 그대로 읽는다. 생성형이 낭비로 손해를 볼 것이라는 기대와 다를 수 있다.
''')
b.code('''
# 셀 6 · 선형 프로브 : 생성형 vs JEPA
t0 = time.time()
for s in SEEDS:
    if s not in GEN:  GEN[s] = train_gen(s)
    if s not in JEPA: JEPA[s] = train_jepa(s)
print(f"시드 {SEEDS} 학습 완료 · 누적 {time.time() - t0:.0f}초")
r2_gen = [probe_r2(GEN[s][0]) for s in SEEDS]
r2_jepa = [JEPA[s][3][-1]["probe_r2"] for s in SEEDS]
r2_init = []
for s in SEEDS:
    seed_all(s); r2_init.append(probe_r2(Encoder().to(DEVICE).eval()))
r2_pix = probe_r2(None, feat=lambda im: im.flatten(1))
tab6 = pd.DataFrame({"프로브 R² (x, y 평균)": {"원본 픽셀 256개에 선형 회귀": f"{r2_pix:.3f}", "학습 전 인코더(무작위)": ms(r2_init, 3),
                                              "생성형 인코더(픽셀 복원)": ms(r2_gen, 3), "JEPA 인코더(특징 예측)": ms(r2_jepa, 3)}})
print("프로브 : 학습 이미지", PROBE_TRAIN, "장 · 시험", PROBE_TEST, "장 · 릿지", RIDGE, "· 시드 3개 평균 ± 표준편차")
# 잡음 띠 처리 : 두 인코더의 특징이 잡음 조각 위치와 물체 조각 위치에서 이미지마다 얼마나 변하는가
with torch.no_grad():
    def band_std(enc):
        f = enc(EV_P).std(0).mean(-1); return float(f[:4].mean()), float(f[4:].mean())
    bs = {"생성형": band_std(GEN[0][0]), "JEPA": band_std(JEPA[0][0])}
print("특징 표준편차(시드 0) — 잡음 띠 조각 / 물체 영역 조각 :", {k: f"{a:.3f} / {b_:.3f}" for k, (a, b_) in bs.items()})

@torch.no_grad()
def jepa_error_share(enc, pred, tgt):
    """JEPA 의 가려진 위치 L1 손실을 잡음 띠(조각 0~3)와 물체 영역으로 나눔 — 셀 4 의 생성형 몫과 같은 측정"""
    y = F.layer_norm(take(tgt(EV_P), EV_MSK), (DIM,))
    e = (pred(enc(EV_P, EV_VIS), EV_MSK) - y).abs().mean(-1)
    noise = (EV_MSK < 4).float()
    return dict(잡음띠_몫=float((e * noise).sum() / e.sum()), 잡음띠_조각L1=float((e * noise).sum() / noise.sum()), 물체영역_조각L1=float((e * (1 - noise)).sum() / (1 - noise).sum()))
sh_j = jepa_error_share(JEPA[0][0], JEPA[0][1], JEPA[0][2])
print(f"JEPA 가려진 위치 L1 손실 중 잡음 띠 몫 {sh_j['잡음띠_몫']:.3f} (면적 몫 {sh_g['가린조각중_잡음띠_비율']:.3f}) · 조각당 L1 : 잡음 띠 {sh_j['잡음띠_조각L1']:.4f} · 물체 영역 {sh_j['물체영역_조각L1']:.4f}")
tab6
''')

b.md('''
### 셀 7 · 붕괴 실험

**왜 이 셀을 도는가** : JEPA 는 정답을 **자기 자신이 만든 특징**으로 삼기 때문에 위험한 지름길이 있다 — 인코더가 **모든 입력에 같은 특징**을 내면 예측도 정답도 같아져 손실이 0 이 된다. 이것이 **표현 붕괴**다. 정상 학습이 이 길로 안 빠지는 이유는 (1) 정답 쪽 **경사 차단**, (2) 정답을 천천히 따라가는 **EMA 타깃**이다. 둘을 하나씩 빼 본다.

| 변종 | 경사 차단 | 타깃 인코더 |
|---|---|---|
| ① 정상 | 있음 | EMA 0.99 |
| ② 경사 차단 제거 | 없음 | 컨텍스트 인코더 그 자체(정답도 학습됨) |
| ③ EMA 0 | 있음 | 매 스텝 컨텍스트 인코더를 그대로 복사 |

**측정하는 것(곡선)** : 학습 중 100스텝마다 ① 손실 ② **특징 표준편차**(같은 조각 위치의 특징이 이미지마다 변하는 정도. 0 으로 가면 붕괴) ③ **유효 차원 수**(특징이 몇 개 축에 퍼져 있는가 — 크기와 무관하게 쪼그라듦을 보는 보조 지표. 1 에 가까우면 한 축으로 몰린 것) ④ **프로브 R²**. 프로브는 특징을 표준화한 뒤 읽으므로 **크기만 작아진 것은 벌하지 않는다** — 정보가 실제로 사라졌는지만 본다.

**읽는 법** : 손실이 낮아져도 표준편차가 0 으로 가고 R² 가 떨어지면 붕괴 — 손실만 보면 「학습 성공」으로 착각하는 경우다. 붕괴가 일어나지 않는 변종이 있다면 그 사실을 그대로 읽는다.
''')
b.code('''
# 셀 7 · 붕괴 실험 (정답 특징에 층 정규화를 씌운 상태에서)
VARIANTS = {"① 정상(EMA 0.99 + 경사 차단)": dict(), "② 경사 차단 제거": dict(stop_grad=False), "③ EMA 0(복사) + 경사 차단": dict(ema=0.0)}
t0 = time.time()
RUNS = {"① 정상(EMA 0.99 + 경사 차단)": {s: JEPA[s][3] for s in SEEDS}}
for nm, kw in list(VARIANTS.items())[1:]:
    RUNS[nm] = {s: train_jepa(s, **kw)[3] for s in SEEDS}
print(f"변종 학습 {time.time() - t0:.0f}초 · 시드 {SEEDS} · 스텝 {STEPS} · 정답 특징 층 정규화 있음")

def curves(runs):
    steps = [r["step"] for r in next(iter(runs.values()))]
    return steps, {k: np.array([[r[k] for r in lg] for lg in runs.values()]) for k in ("loss", "feat_std", "eff_rank", "probe_r2")}
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
for (nm, runs), col in zip(RUNS.items(), ("tab:blue", "tab:red", "tab:green")):
    st, cv = curves(runs)
    for a, k, t in zip(ax, ("loss", "feat_std", "eff_rank", "probe_r2"), ("L1 손실", "특징 표준편차(0 이면 붕괴)", "유효 차원 수(1 이면 한 축)", "프로브 R²")):
        m, sd = cv[k].mean(0), cv[k].std(0); a.plot(st, m, color=col, label=nm); a.fill_between(st, m - sd, m + sd, color=col, alpha=0.2); a.set_title(t); a.set_xlabel("스텝")
ax[0].legend(fontsize=8); logaxis(ax[0]); logaxis(ax[1])
plt.tight_layout(); plt.show()
rows = []
for nm, runs in RUNS.items():
    st, cv = curves(runs)
    rows.append({"변종": nm, "마지막 L1 손실": ms(cv["loss"][:, -1], 4), "마지막 특징 표준편차": ms(cv["feat_std"][:, -1], 4), "마지막 유효 차원": ms(cv["eff_rank"][:, -1], 1), "마지막 프로브 R²": ms(cv["probe_r2"][:, -1], 3)})
tab7 = pd.DataFrame(rows); tab7
''')

b.code('''
# 셀 7-2 · 정상이 아닌 변종이 붕괴하지 않았다면 : 정답 정규화를 빼면 어떻게 되는가
#   (셀 7 의 변종에서 정답 정규화를 빼면 붕괴가 어떻게 달라지는지 보는 실험 — ① 정상은 셀 7 의 것과 같은 설정이라 다시 돌리지 않는다)
VAR2 = {"②' 경사 차단 제거, 정규화 없음": dict(stop_grad=False, target_norm=False), "③' EMA 0, 정규화 없음": dict(ema=0.0, target_norm=False)}
t0 = time.time()
RUNS2 = {nm: {s: train_jepa(s, **kw)[3] for s in SEEDS} for nm, kw in VAR2.items()}
print(f"추가 변종 학습 {time.time() - t0:.0f}초 · 정답 특징 층 정규화 없음")
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
for (nm, runs), col in zip(RUNS2.items(), ("tab:blue", "tab:red", "tab:green")):
    st, cv = curves(runs)
    for a, k, t in zip(ax, ("loss", "feat_std", "eff_rank", "probe_r2"), ("L1 손실", "특징 표준편차(0 이면 붕괴)", "유효 차원 수(1 이면 한 축)", "프로브 R²")):
        m, sd = cv[k].mean(0), cv[k].std(0); a.plot(st, m, color=col, label=nm); a.fill_between(st, m - sd, m + sd, color=col, alpha=0.2); a.set_title(t); a.set_xlabel("스텝")
ax[0].legend(fontsize=8); logaxis(ax[0]); logaxis(ax[1])
plt.tight_layout(); plt.show()
rows = []
for nm, runs in RUNS2.items():
    st, cv = curves(runs)
    rows.append({"변종": nm, "마지막 L1 손실": ms(cv["loss"][:, -1], 4), "마지막 특징 표준편차": ms(cv["feat_std"][:, -1], 4), "마지막 유효 차원": ms(cv["eff_rank"][:, -1], 1), "마지막 프로브 R²": ms(cv["probe_r2"][:, -1], 3)})
tab72 = pd.DataFrame(rows); tab72
''')

b.md('''
### 셀 8 · 정리 · 확인 질문

**왜 이 셀을 도는가** : 핵심 숫자를 한 번에 모으고 `results/measured_02.json` 에 저장한다(장표에 옮길 숫자는 여기서만 인용).
''')
b.code('''
# 셀 8 · 정리
def last(runs, k):
    return [float(lg[-1][k]) for lg in runs.values()]
measured = dict(
    설정=dict(장치=device_info(), 시드=SEEDS, 학습스텝=STEPS, 배치=BATCH, 학습률=LR, 특징차원=DIM, EMA계수=EMA_TAU, 보이는조각=N_VISIBLE, 전체조각=16,
              프로브=dict(학습이미지=PROBE_TRAIN, 시험이미지=PROBE_TEST, 릿지=RIDGE), 평가이미지수=len(EV_IMG)),
    L1손계산=dict(합=hand_sum, 평균=hand_sum / 2, torch_평균=F.l1_loss(pred, tgt).item(), torch_합=F.l1_loss(pred, tgt, reduction="sum").item()),
    생성형_시드0_잡음띠=sh_g, JEPA_시드0_잡음띠=sh_j,
    프로브_R2=dict(원본픽셀=r2_pix, 학습전인코더=r2_init, 생성형=r2_gen, JEPA=r2_jepa),
    특징표준편차_시드0_잡음띠조각_물체조각={k: dict(잡음띠=a, 물체=b_) for k, (a, b_) in bs.items()},
    붕괴실험_정규화있음={nm: dict(마지막_손실=last(r, "loss"), 마지막_특징표준편차=last(r, "feat_std"), 마지막_유효차원=last(r, "eff_rank"), 마지막_프로브R2=last(r, "probe_r2")) for nm, r in RUNS.items()},
    붕괴실험_정규화없음={nm: dict(마지막_손실=last(r, "loss"), 마지막_특징표준편차=last(r, "feat_std"), 마지막_유효차원=last(r, "eff_rank"), 마지막_프로브R2=last(r, "probe_r2")) for nm, r in RUNS2.items()})
save_measured("02", measured)
print(tab6.to_string()); print(); print(tab7.to_string(index=False)); print(); print(tab72.to_string(index=False))
''')

b.md('''
**확인 질문**
1. 생성형 모델의 픽셀 손실에서 잡음 띠가 차지한 몫(셀 4)은 면적 비율과 비교해 어땠는가? 같은 측정을 JEPA 의 L1 손실에 했을 때(셀 6) 몫은 어떠했고, 두 인코더의 잡음 띠 조각 특징 표준편차는 어떠했는가? 이 숫자들이 「JEPA 는 맞힐 수 없는 곳을 덜 쓴다」는 주장을 뒷받침하는가?
2. JEPA 는 화면을 그리지 않는데도 물체 위치가 특징에 들어 있는지를 어떻게 확인했는가? 프로브가 선형이어야 하는 이유는?
3. 경사 차단을 빼거나 EMA 를 0 으로 둔 변종에서 손실 · 특징 표준편차 · 프로브 R² 는 각각 어떻게 움직였는가? 손실만 보면 무엇을 놓치는가? (셀 7 · 셀 7-2 의 숫자로)
''')

b.save()
