"""6-2 실습 노트북 01 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook_01.py   → ../notebooks/01_월드모델과_계획.ipynb (출력 없이)
"""
from nb import Book

b = Book("01_월드모델과_계획.ipynb")

b.md('''
# 실습 01 · 월드모델과 계획 — 냉각수 밸브로 해 보는 「예측해 보고 선택한다」

**직접 해 보는 것** : 냉각수 밸브를 직접 만든 시뮬레이터로 두고, (1) 관측과 상태가 왜 다른지 (2) 월드모델을 학습해 두 행동의 결과를 예측해 비교하는 법 (3) 예측이 길어질수록 틀어지는 누적 오차 (4) 후보 행동열을 선택하는 세 가지 방법(무작위 · CEM · MPPI) (5) 계획을 끝까지 실행하는 것과 매 스텝 다시 계획하는 것(MPC)의 차이를 숫자로 확인한다.

**강의의 개념과 이어지는 곳** : 관측 · 상태 · 행동 · 월드모델 · 비용 · 계획과 정책의 차이 · 누적 오차(drift) · 후보 비교(같은 상태, 두 행동, 두 예측) · CEM · MPPI · MPC(재계획).

| 셀 | 하는 일 |
|---|---|
| 1 | 설정 — 시뮬레이터 계수와 실험 설정을 전부 출력 |
| 2 | 환경 이해 — 식을 손으로 계산한 값 = 시뮬레이터 값, 임의 행동 궤적 그림 |
| 3 | 관측 vs 상태 — 입력을 「현재 관측 1개」 vs 「최근 k개」로 바꿔 1스텝 예측 오차 비교 |
| 4 | 월드모델 학습 곡선 |
| 5 | 후보 비교 — 같은 상태에서 개도 +10% vs −10% |
| 6 | 누적 오차 — 개루프 예측 오차 곡선, 학습 입력에 잡음을 섞은 모델과 비교 |
| 7 | 계획기 비교 — 무작위 최고 vs CEM vs MPPI (같은 예산) |
| 8 | MPC vs 개루프 — 모델이 어긋난 경우 포함 |
| 9 | 선택 — 이산 행동 전수 탐색 vs CEM |
| 10 | 정리 · 확인 질문 3개 |

- 실행 : **Run → Run All Cells**. GPU 없어도 되고(CPU 에서 더 오래 걸림), 외부 데이터 없이 합성 데이터만 쓴다
- 결과를 바꾸는 설정(시드 · 학습 스텝 · 표본 수)은 모두 출력에 찍는다. 결과가 기대와 달라도 그대로 남긴다
''')

b.md('''
### 셀 1 · 설정

**왜 이 셀을 도는가** : 이후 모든 숫자가 어떤 설정에서 나왔는지 한곳에 찍어 둔다. 설정을 바꾸고 다시 돌려 보는 출발점.

시뮬레이터 약속 — 이 장치는 이렇게 움직인다.
- **상태** 온도 T(℃). 열 부하 L 이 데우고, 냉각수 밸브가 식힌다. 열 부하는 천천히 오르내리고(사인 모양 + 작은 잡음) **관측에는 안 보인다**
- **행동** a = 밸브 개도의 변화량(−10 ~ +10 %/스텝). 개도 지령 c 는 0~100 % 로 막힌다
- **지연** 개도 지령을 바꿔도 냉각은 **2스텝 뒤 온도부터** 반영된다(지령 c_s 는 T_(s+2) 부터 영향)
- **관측** o = (잡음 낀 온도, 현재 지령값). 온도 센서 잡음 표준편차 0.5℃
- **비용** (T−70)² + 0.05·a² + 20·max(0, T−85)² — 70℃ 에 가깝게, 개도를 덜 흔들게, 85℃ 상한을 넘지 않게
''')
b.code('''
# 셀 1 · 설정
import sys; sys.path.insert(0, ".")
from wm_common import *
import copy, functools

SEED = 0                    # 기본 시드(모델 학습 시드는 0·1·2 세 개를 씀)
MODEL_SEEDS = (0, 1, 2)
EPOCHS = 40                 # 월드모델 학습 에폭
LR, BATCH = 2e-3, 256       # 학습률 · 배치 크기
HIDDEN = 128                # 은닉 유닛 수(은닉층 2개)
K_HIST = 4                  # 월드모델이 보는 「최근 관측」 개수(기본)
AUG_SD = 1.0                # 잡음 섞기(noise augmentation) 세기 — 입력 온도에 더하는 표준편차(℃)
N_ENV, N_STEP = 200, 100    # 학습 데이터 : 200대 × 100스텝 = 20,000 전이(무작위 행동)

print("장치 :", device_info())
print("시뮬레이터 :", VALVE_CONFIG)
print(f"학습 데이터 {N_ENV}대 × {N_STEP}스텝 = {N_ENV * N_STEP:,} 전이 · 에폭 {EPOCHS} · 학습률 {LR} · 배치 {BATCH} · 은닉 {HIDDEN}x2")
print("월드모델 시드", MODEL_SEEDS, "· 최근 관측 k =", K_HIST, "· 잡음 섞기 표준편차", AUG_SD, "℃")
''')

b.md('''
### 셀 2 · 환경 이해 — 식을 손으로 계산 + 임의 행동 궤적

**왜 이 셀을 도는가** : 월드모델이 배울 대상(시뮬레이터)을 먼저 눈으로 본다. 그리고 시뮬레이터가 위의 식대로 움직이는지 **손으로 계산한 값 = 코드 값**으로 확인한다.

**읽는 법** : 위 그림은 온도(70℃ 가 목표, 85℃ 가 상한), 가운데는 개도 지령, 아래는 열 부하. 개도를 크게 바꿔도 온도가 곧바로 꺾이지 않고 **두 박자 늦게** 움직이는 것을 본다.
''')
b.code('''
# 셀 2 · 환경 이해
# (a) 식을 손으로 계산 : T_next = T + 0.1·L − 0.004·c_prev·(T − 25)  /  비용 = (T−70)² + 0.05a² + 20·max(0,T−85)²
sim = ValveSim(1, 5)
T, c_prev = sim.T[0], sim.c_prev[0]
noise = copy.deepcopy(sim.rng).normal(0, LOAD_NOISE, 1)[0]      # step() 이 곧 뽑을 열 부하 잡음을 미리 엿봄
L = sim.load()[0] + noise
hand = T + 0.1 * L - 0.004 * c_prev * (T - 25)
sim.step(np.array([4.0]))
print(f"T={T:.3f}  L={L:.3f}  c_prev={c_prev:.2f}  →  손 계산 T_next={hand:.4f}  시뮬레이터={sim.T[0]:.4f}  차이={abs(hand - sim.T[0]):.2e}")
hc = (88 - 70) ** 2 + 0.05 * 4 ** 2 + 20 * (88 - 85) ** 2          # 324 + 0.8 + 180
print(f"비용 손 계산(T=88, a=4) = 324 + 0.8 + 180 = {hc:.1f}  함수값 = {step_cost(88.0, 4.0):.1f}")

# (b) 임의 행동 궤적 3개
d = collect_random(3, EP_LEN, 11)
sims = ValveSim(3, 11)
loads = []
for t in range(EP_LEN):
    loads.append(sims.load().copy()); sims.t += 1
loads = np.array(loads).T
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for i in range(3):
    ax[0].plot(d["T_next"][i], label=f"궤적 {i}"); ax[1].plot(d["c"][i]); ax[2].plot(loads[i])
ax[0].axhline(T_TARGET, color="g", ls="--", lw=1); ax[0].axhline(T_LIMIT, color="r", ls="--", lw=1)
ax[0].set_ylabel("온도 T (℃)"); ax[0].legend(ncol=3); ax[0].set_title("임의 행동 궤적 3개 — 초록 = 목표 70℃ · 빨강 = 상한 85℃")
ax[1].set_ylabel("개도 지령 c (%)"); ax[2].set_ylabel("열 부하 L(잡음 제외)"); ax[2].set_xlabel("스텝")
plt.tight_layout(); plt.show()
''')

b.md('''
### 셀 3 · 관측 vs 상태 — 입력을 몇 개 보여 줘야 하는가

**왜 이 셀을 도는가** : 센서가 주는 **관측**은 장치의 **상태**와 다르다. 이 장치는 (1) 온도에 잡음이 끼고 (2) 지금 냉각을 결정하는 것은 **한 스텝 전의 개도 지령**인데 관측에는 현재 지령만 있고 (3) 열 부하는 아예 안 보인다. 그래서 월드모델 입력을 「현재 관측 1개」로 하면 다음 온도를 맞히는 데 한계가 있고, 「최근 k개 관측」을 주면 지난 지령 · 온도 추세(= 열 부하의 흔적)가 들어와 상태에 더 가까워진다.

**하는 일** : 무작위 행동 데이터 20,000 전이로 입력 k = 1·2·4·8 인 모델을 시드 3개씩 학습하고, 따로 만든 시험 데이터(50대 × 100스텝)에서 **다음 온도(잡음 없는 참값)**를 얼마나 맞히는지 RMSE(℃)로 비교한다. 비교 기준 「변화 없음」은 T_(t+1) = 지금 관측한 T 로 찍는 것.

**읽는 법** : 표의 RMSE 가 작을수록 좋다. k=1 → k=2 에서 가장 크게 줄면 「한 스텝 전 지령」이 핵심이었다는 뜻.
''')
b.code('''
# 셀 3 · 관측 vs 상태
TRAIN = collect_random(N_ENV, N_STEP, seed=0)
TEST = collect_random(50, N_STEP, seed=999)
print("학습 데이터 온도 범위 :", TRAIN["T_next"].min().round(1), "~", TRAIN["T_next"].max().round(1), "℃ · 개도 범위", TRAIN["c"].min().round(1), "~", TRAIN["c"].max().round(1), "%")

def fit(k, seed, aug=0.0, epochs=EPOCHS, data=TRAIN):
    """월드모델 학습. 입력 = 최근 k개 관측(+행동), 정답 = 다음 온도(참값). aug>0 이면 입력 온도에 잡음을 섞는다"""
    seed_all(seed)
    x, y = make_windows(data, k); xt, yt = make_windows(TEST, k)
    X, Y = torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)
    Xt, Yt = torch.tensor(xt, device=DEVICE), torch.tensor(yt, device=DEVICE)
    m = mlp(2 * k + 1, 1, HIDDEN).to(DEVICE); opt = torch.optim.Adam(m.parameters(), LR)
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=DEVICE); tr = 0.0
        for i in range(0, len(X), BATCH):
            ix = perm[i:i + BATCH]; xb = X[ix].clone()
            if aug > 0:
                xb[:, :k] += torch.randn_like(xb[:, :k]) * aug / 10        # 온도 입력 열에만(정규화 단위 = ℃/10)
            loss = F.mse_loss(m(xb), Y[ix]); opt.zero_grad(); loss.backward(); opt.step(); tr += loss.item() * len(ix)
        with torch.no_grad():
            curve.append((math.sqrt(tr / len(X)) * 3, F.mse_loss(m(Xt), Yt).sqrt().item() * 3))
    return m.eval(), np.array(curve)

t0 = time.time()
MODELS, CURVES = {}, {}
for k in (1, 2, 4, 8):
    for s in MODEL_SEEDS:
        MODELS[(k, 0.0, s)], CURVES[(k, 0.0, s)] = fit(k, s)
print(f"학습 {len(MODELS)}개 모델 {time.time() - t0:.0f}초")

x1, y1 = make_windows(TEST, 1)                                    # 「변화 없음」 기준선 : 예측 변화량 0
naive = float(np.sqrt(np.mean(y1 ** 2)) * 3)
rows = [dict(입력="변화 없음(T_next = 관측 T)", **{"1스텝 RMSE(℃)": f"{naive:.3f}"})]
rmse1 = {}
for k in (1, 2, 4, 8):
    v = [CURVES[(k, 0.0, s)][-1, 1] for s in MODEL_SEEDS]; rmse1[k] = v
    rows.append(dict(입력=f"최근 관측 {k}개", **{"1스텝 RMSE(℃)": ms(v, 3)}))
pd.DataFrame(rows)
''')

b.md('''
### 셀 4 · 월드모델 학습 곡선

**왜 이 셀을 도는가** : 월드모델 학습의 정답은 **실제로 다음에 관측된 값**이다(「이 상태에서 이 행동을 하면 다음엔 이렇게 된다」를 맞히기). 학습이 제대로 수렴했는지, 입력 1개와 4개의 차이가 학습 내내 벌어져 있는지 본다.

**읽는 법** : 실선 = 학습 데이터 오차, 점선 = 시험 데이터 오차(시드 3개 평균). 두 선이 크게 벌어지면 외우기(과적합), 나란히 내려가면 정상.
''')
b.code('''
# 셀 4 · 학습 곡선
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for k, col in ((1, "tab:red"), (4, "tab:blue")):
    tr = np.mean([CURVES[(k, 0.0, s)][:, 0] for s in MODEL_SEEDS], 0); te = np.mean([CURVES[(k, 0.0, s)][:, 1] for s in MODEL_SEEDS], 0)
    ax.plot(tr, color=col, label=f"최근 관측 {k}개 · 학습"); ax.plot(te, color=col, ls="--", label=f"최근 관측 {k}개 · 시험")
ax.set_xlabel("에폭"); ax.set_ylabel("다음 온도 RMSE (℃)"); logaxis(ax); ax.legend(); ax.set_title("월드모델 학습 곡선 (시드 3개 평균)")
plt.tight_layout(); plt.show()
for k in (1, 4):
    print(f"k={k} 마지막 에폭 : 학습 {np.mean([CURVES[(k, 0.0, s)][-1, 0] for s in MODEL_SEEDS]):.3f} ℃ · 시험 {np.mean([CURVES[(k, 0.0, s)][-1, 1] for s in MODEL_SEEDS]):.3f} ℃")
''')

b.md('''
### 셀 5 · 후보 비교 — 같은 상태, 두 행동, 두 예측

**왜 이 셀을 도는가** : 월드모델을 쓰는 가장 작은 형태. **지금 같은 상태에서** 행동 후보 두 개(개도를 +10% 올린다 / −10% 내린다)를 각각 모델에 넣어 앞으로 12스텝의 온도를 **예측**하고, 예측한 온도로 비용을 계산해 더 싼 쪽을 선택한다. 실제로 해 보지 않고 머릿속에서 먼저 해 보는 것.

**약속** : 첫 스텝에만 ±10% 를 쓰고 이후는 개도 변화 0(그대로 둔다). 열 부하가 높아 온도가 오르는 상황을 일부러 골랐다. 실선 = 모델 예측, 점선 = 같은 잡음을 쓴 실제 시뮬레이터에서의 결과.

**읽는 법** : 개도를 바꿔도 처음 두 스텝은 두 예측이 같고(지연), 그 뒤 갈라진다. 비용 표에서 모델이 선택한 쪽이 실제 비용이 더 싼 쪽과 같은지 본다.
''')
b.code('''
# 셀 5 · 후보 비교
H5 = 12
sim = ValveSim(1, 21, T0=(76, 76.01), c0=(50, 50.01), load_base=(106, 106.01), load_amp=(5, 5.01))
Ts, Cs = [], []
for _ in range(K_HIST + 2):                                        # 개도 그대로 두고 몇 스텝 흘려 최근 관측 쌓기
    o = sim.obs(); Ts.append(o[0][0]); Cs.append(o[1][0]); sim.step(np.array([0.0]))
def hist_t(Ts, Cs, k):
    T, C = list(Ts[-k:]), list(Cs[-k:])
    while len(T) < k:
        T, C = [T[0]] + T, [C[0]] + C
    return torch.tensor([T], dtype=torch.float32, device=DEVICE), torch.tensor([C], dtype=torch.float32, device=DEVICE)
Th, Ch = hist_t(Ts, Cs, K_HIST)
print("지금 관측한 최근 온도", np.round(Ts[-K_HIST:], 1), "· 지령", np.round(Cs[-K_HIST:], 1))

A = torch.zeros(2, H5, device=DEVICE); A[0, 0], A[1, 0] = 10.0, -10.0
model = MODELS[(K_HIST, 0.0, 0)]
Tp = rollout(model, Th.expand(2, -1), Ch.expand(2, -1), A)
J_pred = plan_cost(Tp, A).cpu().numpy()
real, J_real = [], []
for a0 in (10.0, -10.0):                                           # 같은 시뮬레이터 상태 · 같은 잡음으로 실제 실행
    s2 = copy.deepcopy(sim); Tr, cost = [], 0.0
    for h in range(H5):
        a = a0 if h == 0 else 0.0
        s2.step(np.array([a])); Tr.append(s2.T[0]); cost += step_cost(s2.T[0], a)
    real.append(Tr); J_real.append(cost)
Tp = Tp.cpu().numpy()
fig, ax = plt.subplots(figsize=(8, 4))
for i, (nm, col) in enumerate((("개도 +10%", "tab:red"), ("개도 −10%", "tab:blue"))):
    ax.plot(range(1, H5 + 1), Tp[i], color=col, label=f"{nm} · 모델 예측"); ax.plot(range(1, H5 + 1), real[i], color=col, ls="--", label=f"{nm} · 실제")
ax.axhline(T_TARGET, color="g", lw=1, ls=":"); ax.set_xlabel("몇 스텝 뒤"); ax.set_ylabel("온도 (℃)"); ax.legend(); ax.set_title("같은 상태에서 행동 두 개 — 예측 온도열")
plt.tight_layout(); plt.show()
print(pd.DataFrame({"행동": ["개도 +10%", "개도 −10%"], "예측 비용": J_pred.round(1), "실제 비용": np.round(J_real, 1)}).to_string(index=False))
print("모델이 선택한 쪽 :", ["개도 +10%", "개도 −10%"][int(J_pred.argmin())], "· 실제로 싼 쪽 :", ["개도 +10%", "개도 −10%"][int(np.argmin(J_real))])
case5 = dict(예측비용_개도올림=float(J_pred[0]), 예측비용_개도내림=float(J_pred[1]), 실제비용_개도올림=float(J_real[0]), 실제비용_개도내림=float(J_real[1]))
''')

b.md('''
### 셀 6 · 누적 오차 — 모델만으로 멀리 예측하면

**왜 이 셀을 도는가** : 월드모델로 먼 미래를 보려면 **예측한 온도를 다시 입력으로 넣어** 한 스텝씩 이어 가야 한다. 한 스텝의 작은 오차가 다음 입력에 섞여 쌓인다 — 이것이 누적 오차(drift)다. 학습 때는 입력이 늘 「실제 관측(잡음 포함)」인데 예측을 이어 갈 때는 「모델 자신의 예측」이 들어오는 불일치도 한몫한다.

**하는 일** : 시험 데이터의 650개 지점에서 최근 관측과 **그 뒤 실제로 쓰인 행동열**을 모델에 주고, H = 1·5·10·20·40 스텝 뒤 온도 예측 오차(RMSE)를 측정한다. 비교 대상 셋.
- 보통 모델 / **잡음 섞기 모델**(학습할 때 입력 온도에 표준편차 1℃ 잡음을 더해, 입력이 완벽하지 않아도 버티게 한 모델)
- **물리를 정확히 아는 모델**(바닥 기준) : 시뮬레이터 식 그대로지만 열 부하 잡음은 미리 알 수 없어 다른 잡음으로 돌린 것 — 어떤 모델도 이 아래로는 못 내려간다(예측 불가능한 흔들림의 크기)

**읽는 법** : H=1 값이 셀 3 의 1스텝 RMSE 와 조금 다른 것은 평가 지점이 다르기 때문(여기서는 시작 직후 4스텝을 뺌). 오차 곡선이 H 가 커질수록 올라가는 모양, 그리고 바닥 기준과의 간격. 잡음 섞기가 이득인지 손해인지는 결과 그대로 읽는다.
''')
b.code('''
# 셀 6 · 누적 오차
t0 = time.time()
for s in MODEL_SEEDS:
    MODELS[(K_HIST, AUG_SD, s)], CURVES[(K_HIST, AUG_SD, s)] = fit(K_HIST, s, aug=AUG_SD)
print(f"잡음 섞기 모델 {len(MODEL_SEEDS)}개 학습 {time.time() - t0:.0f}초 (표준편차 {AUG_SD}℃) · 1스텝 시험 RMSE :",
      ms([CURVES[(K_HIST, AUG_SD, s)][-1, 1] for s in MODEL_SEEDS], 3), "℃ (보통 모델", ms(rmse1[K_HIST], 3), "℃)")

HS = (1, 5, 10, 20, 40); HMAX = 40
T0S = list(range(K_HIST, 55, 4))
TESTK = collect_random(50, N_STEP, seed=999, keep_sims=True)
def open_loop_err(model):
    err = {h: [] for h in HS}
    for t0 in T0S:
        sl = slice(t0 - K_HIST + 1, t0 + 1)
        Th = torch.tensor(TESTK["T_obs"][:, sl], dtype=torch.float32, device=DEVICE); Ch = torch.tensor(TESTK["c"][:, sl], dtype=torch.float32, device=DEVICE)
        A = torch.tensor(TESTK["a"][:, t0:t0 + HMAX], dtype=torch.float32, device=DEVICE)
        P = rollout(model, Th, Ch, A).cpu().numpy(); truth = TESTK["T_next"][:, t0:t0 + HMAX]
        for h in HS:
            err[h].append((P[:, h - 1] - truth[:, h - 1]) ** 2)
    return {h: float(np.sqrt(np.mean(err[h]))) for h in HS}
def floor_err():
    err = {h: [] for h in HS}
    for t0 in T0S:
        s2 = copy.deepcopy(TESTK["sims"][t0]); s2.rng = np.random.default_rng(777 + t0)           # 다른 잡음으로 같은 물리를 다시 돌림
        truth = TESTK["T_next"][:, t0:t0 + HMAX]
        for h in range(1, HMAX + 1):
            s2.step(TESTK["a"][:, t0 + h - 1])
            if h in HS:
                err[h].append((s2.T - truth[:, h - 1]) ** 2)
    return {h: float(np.sqrt(np.mean(err[h]))) for h in HS}

E_plain = [open_loop_err(MODELS[(K_HIST, 0.0, s)]) for s in MODEL_SEEDS]
E_aug = [open_loop_err(MODELS[(K_HIST, AUG_SD, s)]) for s in MODEL_SEEDS]
E_floor = floor_err()
tab = pd.DataFrame({"H(스텝)": HS,
                    "보통 모델": [ms([e[h] for e in E_plain], 2) for h in HS],
                    "잡음 섞기 모델": [ms([e[h] for e in E_aug], 2) for h in HS],
                    "바닥 기준(물리 정확, 잡음 모름)": [f"{E_floor[h]:.2f}" for h in HS]})
print(f"평가 지점 {len(T0S) * 50}개 · 단위 ℃ (시드 3개 평균 ± 표준편차)")
fig, ax = plt.subplots(figsize=(8, 4))
for E, nm, col in ((E_plain, "보통 모델", "tab:blue"), (E_aug, "잡음 섞기 모델", "tab:orange")):
    m = np.array([[e[h] for h in HS] for e in E]); ax.errorbar(HS, m.mean(0), m.std(0), marker="o", color=col, label=nm, capsize=3)
ax.plot(HS, [E_floor[h] for h in HS], "k--", marker="s", label="바닥 기준")
ax.set_xlabel("예측 스텝 수 H"); ax.set_ylabel("온도 예측 RMSE (℃)"); logaxis(ax, x=True, y=True); ax.legend(); ax.set_title("모델만으로 멀리 예측하면 오차가 쌓인다")
plt.tight_layout(); plt.show()
tab
''')

b.md('''
### 셀 7 · 계획기 비교 — 무작위 최고 vs CEM vs MPPI

**왜 이 셀을 도는가** : 월드모델이 「행동열 → 예측 비용」을 줄 수 있으면, 남은 일은 **비용이 가장 낮은 행동열을 찾는 것**이다. 모델 안에서의 시도는 실제 장치를 건드리지 않으니 싸다. 세 방법을 **같은 예산**(한 번 계획에 표본 256개 × 4번 반복 = 모델 롤아웃 1,024개)으로 비교한다.

- **무작위 최고** : 1,024개를 한 번에 무작위로 뽑아 비용이 가장 낮은 것을 쓴다
- **CEM(교차 엔트로피 방법)** : 정규분포에서 행동열 256개를 뽑아 비용을 매기고, **상위 K = 32개(엘리트)의 평균과 표준편차**로 분포를 갱신. 이를 4번 반복해 분포가 좋은 쪽으로 좁아진다
- **MPPI(모델 예측 경로 적분)** : 같은 방식으로 뽑되, 갱신은 **비용이 낮을수록 큰 가중치**(exp(−(비용−최소)/온도))로 전체의 가중 평균. 엘리트를 자르지 않고 부드럽게 반영

계획 길이 H = 12 스텝. 계획은 매 스텝 다시 한다(MPC). 이전 계획을 한 칸 밀어 다음 계획의 시작점으로 쓴다(웜스타트 — 무작위는 해당 없음). 에피소드 80스텝, 비용은 **한 스텝 평균 비용**.

**읽는 법** : 평균 ± 표준편차는 에피소드 9개(모델 시드 3 × 에피소드 3). 계획 시간은 한 번 계획에 걸린 평균(ms).
''')
b.code('''
# 셀 7 · 계획기 비교 (CEM · MPPI 는 손으로 구현)
def eval_plans(model, Th, Ch, A):
    N = A.shape[0]
    return plan_cost(rollout(model, Th.expand(N, -1), Ch.expand(N, -1), A), A)

def pad(A, H):                                        # 앞쪽 nfree 개만 자유, 나머지 스텝은 개도 변화 0
    return A if A.shape[-1] == H else torch.cat([A, A.new_zeros(*A.shape[:-1], H - A.shape[-1])], -1)

def plan_random(model, Th, Ch, H, g, m0=None, N=256, iters=4):
    A = (torch.rand(N * iters, H, generator=g, device=DEVICE) * 2 - 1) * A_MAX
    return A[eval_plans(model, Th, Ch, A).argmin()]

def plan_cem(model, Th, Ch, H, g, m0=None, N=256, iters=4, K=32, std0=5.0, nfree=None):
    nf = nfree or H
    mean = torch.zeros(nf, device=DEVICE) if m0 is None else m0[:nf].clone()
    std = torch.full((nf,), std0, device=DEVICE)
    for _ in range(iters):
        A = (mean + std * torch.randn(N, nf, generator=g, device=DEVICE)).clamp(-A_MAX, A_MAX)
        J = eval_plans(model, Th, Ch, pad(A, H))
        elite = A[J.topk(K, largest=False).indices]             # 비용 낮은 상위 K 개
        mean, std = elite.mean(0), elite.std(0) + 0.3           # 평균 · 표준편차 갱신(표준편차 바닥 0.3 : 너무 일찍 좁아지지 않게)
    return pad(mean, H)

def plan_mppi(model, Th, Ch, H, g, m0=None, N=256, iters=4, sigma=5.0, beta=0.1):
    mean = torch.zeros(H, device=DEVICE) if m0 is None else m0.clone()
    for _ in range(iters):
        A = (mean + sigma * torch.randn(N, H, generator=g, device=DEVICE)).clamp(-A_MAX, A_MAX)
        J = eval_plans(model, Th, Ch, A)
        w = torch.softmax(-(J - J.min()) / (beta * J.std() + 1e-6), 0)    # 비용이 낮을수록 가중치 큼(온도 = beta × 비용 표준편차)
        mean = (w[:, None] * A).sum(0)                                      # 가중 평균
    return mean

def run_episode(model, k, plan_fn, ep_seed, H, mode="mpc", sim_kw=None, steps=EP_LEN):
    """mode='mpc' : 매 스텝 계획하고 첫 행동만 실행 / 'open' : 계획한 H 스텝을 끝까지 실행한 뒤 다시 계획"""
    sim = ValveSim(1, ep_seed, T0=(72, 78), c0=(40, 60), **(sim_kw or {}))
    g = torch.Generator(device=DEVICE); g.manual_seed(ep_seed)
    o = sim.obs(); Ts, Cs = [o[0][0]], [o[1][0]]
    costs, Tt, queue, prev, pms = [], [], [], None, []
    for t in range(steps):
        if plan_fn is None:
            a = 0.0
        else:
            if mode == "mpc" or not queue:
                Th, Ch = hist_t(Ts, Cs, k)
                t1 = time.perf_counter()
                m0 = None if (prev is None or mode != "mpc") else torch.cat([prev[1:], prev.new_zeros(1)])
                plan = plan_fn(model, Th, Ch, H, g, m0)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                pms.append((time.perf_counter() - t1) * 1000); prev = plan; queue = plan.cpu().numpy().tolist()
            a = queue.pop(0)
            if mode == "mpc":
                queue = []
        a = float(np.clip(a, -A_MAX, A_MAX)); sim.step(np.array([a]))
        costs.append(step_cost(sim.T[0], a)); Tt.append(sim.T[0])
        o = sim.obs(); Ts.append(o[0][0]); Cs.append(o[1][0])
    Tt = np.array(Tt)
    return dict(cost=float(np.mean(costs)), rms=float(np.sqrt(np.mean((Tt - T_TARGET) ** 2))), tmax=float(Tt.max()),
                over=float(np.mean(Tt > T_LIMIT)), ms=float(np.mean(pms)) if pms else 0.0)

H7, N7, IT7 = 12, 256, 4
PLANNERS = {"개도 고정(계획 없음)": None, "무작위 최고": plan_random, "CEM": plan_cem, "MPPI": plan_mppi}
EP_SEEDS = {s: [100 + 10 * s + j for j in range(3)] for s in MODEL_SEEDS}
print(f"계획 길이 H={H7} · 표본 {N7} × 반복 {IT7} = 롤아웃 {N7 * IT7}개/계획 · CEM 엘리트 K=32 · MPPI 온도 0.1×비용표준편차 · 에피소드 {EP_LEN}스텝 × 9개")
R7, t0 = {}, time.time()
for nm, fn in PLANNERS.items():
    R7[nm] = [run_episode(MODELS[(K_HIST, 0.0, s)], K_HIST, fn, e, H7) for s in MODEL_SEEDS for e in EP_SEEDS[s]]
print(f"실행 {time.time() - t0:.0f}초")
tab7 = pd.DataFrame({nm: dict(**{"한 스텝 평균 비용": ms([r["cost"] for r in rs], 1), "온도 RMS 오차(℃)": ms([r["rms"] for r in rs], 2),
                                   "최고 온도(℃)": ms([r["tmax"] for r in rs], 1), "상한 초과 비율": f"{np.mean([r['over'] for r in rs]):.3f}",
                                   "계획 1회(ms)": f"{np.mean([r['ms'] for r in rs]):.1f}"}) for nm, rs in R7.items()}).T
tab7
''')

b.md('''
### 셀 8 · MPC vs 개루프 — 모델이 틀려도 다시 계획하면

**왜 이 셀을 도는가** : 계획을 한 번 세우고 끝까지 실행하는 **개루프**는 모델이 정확할 때만 통한다. **MPC(모델 예측 제어)**는 첫 행동 하나만 실행하고 새 관측으로 다시 계획한다 — 모델이 어긋나도 매 스텝 현실로 되돌아오는 길이 있다.

**실험** : 같은 CEM 계획기(H = 20, 표본 256 × 4회)를 두 방식으로 실행.
- **개루프** : 20스텝짜리 계획을 끝까지 실행하고 새로 계획(80스텝에 4번만 계획)
- **MPC** : 매 스텝 계획하고 첫 행동만 실행
- 환경 두 가지 : 학습 때와 같은 열 부하(기준 80~100) / **어긋난 열 부하**(기준 110~125, 진폭 35~45 — 모델이 학습 때 본 적 없는 영역)

**읽는 법** : 모델이 맞는 환경에서도 차이가 있는지, 어긋난 환경에서 차이가 더 벌어지는지. 어긋난 환경에서 모델의 1스텝 오차도 함께 찍어 「얼마나 어긋났는가」를 본다.
''')
b.code('''
# 셀 8 · MPC vs 개루프
H8 = 20
SHIFT = dict(load_base=(110, 125), load_amp=(35, 45))
fn8 = functools.partial(plan_cem, N=256, iters=4, K=32)
TEST_SH = collect_random(50, N_STEP, seed=998, **SHIFT)
xs, ys = make_windows(TEST_SH, K_HIST)
with torch.no_grad():
    sh_err = [F.mse_loss(MODELS[(K_HIST, 0.0, s)](torch.tensor(xs, device=DEVICE)), torch.tensor(ys, device=DEVICE)).sqrt().item() * 3 for s in MODEL_SEEDS]
print(f"계획기 CEM · H={H8} · 표본 256 × 반복 4 · 에피소드 9개(모델 시드 3 × 3)")
print("모델의 1스텝 RMSE : 학습과 같은 열 부하", ms(rmse1[K_HIST], 3), "℃ · 어긋난 열 부하", ms(sh_err, 3), "℃")
R8, t0 = {}, time.time()
for env_nm, kw in (("같은 열 부하", {}), ("어긋난 열 부하", SHIFT)):
    for mode, mnm in (("open", "개루프"), ("mpc", "MPC")):
        R8[(env_nm, mnm)] = [run_episode(MODELS[(K_HIST, 0.0, s)], K_HIST, fn8, e + 1000, H8, mode=mode, sim_kw=kw) for s in MODEL_SEEDS for e in EP_SEEDS[s]]
    R8[(env_nm, "개도 고정")] = [run_episode(None, K_HIST, None, e + 1000, H8, sim_kw=kw) for s in MODEL_SEEDS for e in EP_SEEDS[s]]
print(f"실행 {time.time() - t0:.0f}초")
tab8 = pd.DataFrame([dict(환경=e, 방식=m, **{"한 스텝 평균 비용": ms([r["cost"] for r in rs], 1), "온도 RMS 오차(℃)": ms([r["rms"] for r in rs], 2),
                                              "최고 온도(℃)": ms([r["tmax"] for r in rs], 1), "상한 초과 비율": f"{np.mean([r['over'] for r in rs]):.3f}"})
                     for (e, m), rs in R8.items()])
tab8
''')

b.md('''
### 셀 9 · 선택 — 이산 행동 전수 탐색 vs CEM

**왜 이 셀을 도는가** : 행동이 {−5, 0, +5}처럼 몇 개뿐이고 앞 4스텝만 바꾼다면 후보는 3⁴ = 81개뿐이라 **전부 해 볼 수 있다**. 이것이 트리 탐색의 가장 작은 형태(가지가 3개씩 4단계)다. 후보가 폭발하는 문제에서는 못 쓰지만, 작을 때는 가장 확실한 기준선이 된다.

**하는 일** : 앞 4스텝만 자유롭게 정하고 나머지(총 12스텝까지)는 개도 변화 0으로 둔다.
- **전수 탐색** : 81개를 전부 모델로 평가해 최저 비용을 쓴다(연속 행동이 아니라 −5·0·+5 만)
- **CEM(같은 평가 횟수)** : 연속 행동 4개, 표본 27 × 3번 = 평가 81번
- **CEM(평가 1,024번)** : 표본 256 × 4번

둘 다 MPC 로 실행(매 스텝 계획). 에피소드 9개.

**읽는 법** : 같은 평가 횟수(81)에서 이산 전수가 이기는지 연속 CEM 이 이기는지, 평가를 늘리면 얼마나 좋아지는지.
''')
b.code('''
# 셀 9 · 전수 탐색 vs CEM
import itertools
H9 = 12
GRID = torch.tensor(list(itertools.product((-5.0, 0.0, 5.0), repeat=4)), device=DEVICE)       # (81, 4)
def plan_tree(model, Th, Ch, H, g, m0=None):
    A = pad(GRID, H)
    return A[eval_plans(model, Th, Ch, A).argmin()]
cands = {"전수 탐색 {-5,0,5}^4 (평가 81)": plan_tree,
         "CEM 연속 · 평가 81 (27×3)": functools.partial(plan_cem, N=27, iters=3, K=4, nfree=4),
         "CEM 연속 · 평가 1,024 (256×4)": functools.partial(plan_cem, N=256, iters=4, K=32, nfree=4)}
print(f"앞 4스텝만 자유 · 총 {H9}스텝 · 전수 후보 {len(GRID)}개 · 에피소드 9개")
R9, t0 = {}, time.time()
for nm, fn in cands.items():
    R9[nm] = [run_episode(MODELS[(K_HIST, 0.0, s)], K_HIST, fn, e + 2000, H9) for s in MODEL_SEEDS for e in EP_SEEDS[s]]
print(f"실행 {time.time() - t0:.0f}초")
tab9 = pd.DataFrame({nm: dict(**{"한 스텝 평균 비용": ms([r["cost"] for r in rs], 1), "온도 RMS 오차(℃)": ms([r["rms"] for r in rs], 2),
                                   "최고 온도(℃)": ms([r["tmax"] for r in rs], 1), "계획 1회(ms)": f"{np.mean([r['ms'] for r in rs]):.1f}"}) for nm, rs in R9.items()}).T
tab9
''')

b.md('''
### 셀 10 · 정리 · 확인 질문

**왜 이 셀을 도는가** : 핵심 숫자를 한 표로 모으고 `results/measured_01.json` 에 저장한다(장표에 옮길 숫자는 여기서만 인용).
''')
b.code('''
# 셀 10 · 정리
def pack(rs):
    return dict(비용_평균=float(np.mean([r["cost"] for r in rs])), 비용_표준편차=float(np.std([r["cost"] for r in rs])), 온도RMS=float(np.mean([r["rms"] for r in rs])),
                최고온도=float(np.mean([r["tmax"] for r in rs])), 계획1회_ms=float(np.mean([r["ms"] for r in rs])), 에피소드수=len(rs))
measured = dict(
    설정=dict(장치=device_info(), 시뮬레이터=VALVE_CONFIG, 학습전이수=N_ENV * N_STEP, 에폭=EPOCHS, 학습률=LR, 배치=BATCH, 은닉=f"{HIDDEN}x2",
              모델시드=MODEL_SEEDS, 최근관측k=K_HIST, 잡음섞기_표준편차=AUG_SD, 평가지점수=len(T0S) * 50,
              계획기=dict(H=H7, 표본=N7, 반복=IT7, CEM엘리트K=32, MPPI온도=0.1, MPC비교H=H8, 에피소드스텝=EP_LEN, 에피소드수="9(모델 시드 3 x 3)")),
    관측vs상태_1스텝RMSE_도=dict(변화없음=naive, **{f"최근관측{k}개": dict(평균=float(np.mean(v)), 표준편차=float(np.std(v))) for k, v in rmse1.items()}),
    후보비교_셀5=case5,
    누적오차_RMSE_도=dict(H=list(HS), 보통모델_평균=[float(np.mean([e[h] for e in E_plain])) for h in HS], 보통모델_표준편차=[float(np.std([e[h] for e in E_plain])) for h in HS],
                      잡음섞기_평균=[float(np.mean([e[h] for e in E_aug])) for h in HS], 잡음섞기_표준편차=[float(np.std([e[h] for e in E_aug])) for h in HS],
                      바닥기준=[E_floor[h] for h in HS]),
    계획기비교_한스텝평균비용={nm: pack(rs) for nm, rs in R7.items()},
    MPC_vs_개루프={f"{e}|{m}": pack(rs) for (e, m), rs in R8.items()},
    어긋난환경_모델1스텝RMSE_도=dict(평균=float(np.mean(sh_err)), 표준편차=float(np.std(sh_err))),
    전수탐색_vs_CEM={nm: pack(rs) for nm, rs in R9.items()})
save_measured("01", measured)
print("핵심 숫자 (한 스텝 평균 비용, 낮을수록 좋음)")
print(pd.DataFrame({"계획기 비교(셀 7)": {nm: ms([r["cost"] for r in rs], 1) for nm, rs in R7.items()}}).to_string())
print(pd.DataFrame({"MPC vs 개루프(셀 8)": {f"{e}|{m}": ms([r["cost"] for r in rs], 1) for (e, m), rs in R8.items()}}).to_string())
print(pd.DataFrame({"전수 vs CEM(셀 9)": {nm: ms([r["cost"] for r in rs], 1) for nm, rs in R9.items()}}).to_string())
''')

b.md('''
**확인 질문**
1. 월드모델 입력을 현재 관측 1개에서 최근 k개로 늘리면 다음 온도 예측이 좋아졌다. 이 장치에서 관측이 상태가 아닌 이유 세 가지를 들고, 그중 가장 크게 효과를 낸 것은 무엇이었는지 셀 3 의 표로 설명하라.
2. 누적 오차 표에서 H 가 커질수록 오차가 커지는데, 「바닥 기준」도 같이 커진다. 모델이 완벽해도 줄일 수 없는 오차는 어디서 오는가? 그렇다면 MPC 처럼 매 스텝 다시 계획하는 것이 이 문제를 어떻게 줄이는가?
3. CEM 과 MPPI 는 둘 다 「모델로 해 본 비용」으로 분포를 고쳐 나간다. 두 방법의 갱신 방식 차이(엘리트 평균 vs 비용 가중 평균)를 한 문장으로 말하고, 셀 7 의 같은 예산 비교에서 무작위 최고와 어떻게 달랐는지 숫자로 답하라.
''')

b.save()
