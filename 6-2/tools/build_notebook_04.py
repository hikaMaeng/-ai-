"""6-2 실습 노트북 04 생성기 (강사용).

    python build_notebook_04.py   → ../notebooks/04_변화코드_발견과_행동조건화.ipynb (출력 없이)
"""
from nb import Book

b = Book("04_변화코드_발견과_행동조건화.ipynb")

b.md('''
# 실습 04 · 변화 코드 발견과 행동 조건화 — Genie 축소판

**직접 해 보는 것** : 행동 이름표가 없는 영상(프레임 쌍)만 보고 「프레임 사이의 변화」를 소수의 이산 코드로 배우게 한 뒤, 그 코드가 무엇이었는지 측정하고, 코드로 장면을 움직여 보고, 행동을 모델에 넣는 세 가지 방식을 비교하고, 마지막으로 토큰을 여러 번에 나눠 채우는 디코딩을 해 본다.

**강의의 어느 개념과 이어지는가**
- 행동 라벨 없는 영상에서 학습하는 **잠재 행동 모델** — 변화를 이산 코드(VQ)로 압축
- 코드는 학습이 정한 것이지 **사람이 정한 버튼이 아니다** — 이름은 나중에 사람이 붙인다
- 변화 코드 + 이전 프레임 → 다음 프레임 : **동역학**
- 행동을 모델에 넣는 법 : 무시 · 이어 붙임(concat) · **FiLM / AdaLN**
- 마스크 토큰을 확신 높은 것부터 여러 번에 나눠 채우는 **반복 마스크 채우기(MaskGIT)**

**장난감 세계** : 32×32 화면 위에서 4×4 색 사각형이 한 번에 2픽셀씩 움직인다. 숨은 진짜 행동은 5가지(정지 · 상 · 하 · 좌 · 우)이고 벽에서는 못 나간다. 배경에는 행동과 상관없이 프레임마다 깜빡이는 약한 점 3개가 있다. 학습에는 (프레임 t, 프레임 t+1) 쌍 5만 개만 쓰고, **진짜 행동은 평가에만 쓴다**(셀 4 에서 코드로 확인).

| 셀 | 하는 일 |
|---|---|
| 1 | 설정 |
| 2 | 환경과 프레임 쌍 보기 |
| 3 | VQ 를 손으로 — 가장 가까운 코드로 치환 · 직통 경사 |
| 4 | 잠재 행동 모델(LAM) 학습 — 코드 개수 K = 2 · 5 · 8 |
| 5 | 코드의 정체 측정 — 혼동행렬 · 순도 · 상호정보량 · 안 쓰이는 코드 |
| 6 | 코드에 이름 붙이기와 플레이 — 코드열로 10스텝 롤아웃 |
| 7 | 행동 조건화 세 방식 비교 — 무시 · concat · FiLM |
| 8 | 프레임을 4×4 토큰으로 — 작은 VQ 토크나이저 |
| 9 | 반복 마스크 채우기 — 한 번에 vs 여러 번에 나눠 |
| 10 | 정리 · 결과 저장 · 확인 질문 |

- 실행 : GPU 있으면 약 5분, 없으면 CPU 로도 돌아가지만 시드 · 스텝을 줄인 설정으로 돈다(셀 1 이 어느 설정인지 출력)
- 결과를 바꾸는 설정(시드 · 스텝 · 학습률 · 코드 개수)은 모두 출력에 찍는다. 숫자는 매번 실행 출력이 기준이다
''')

b.md('''
### 셀 1 · 설정
- 시드를 3개 돌려 평균±표준편차로 본다. 학습은 매번 조금씩 다른 곳에 도착하기 때문에 한 번의 결과만 믿지 않기 위해서다
- 학습률은 1e-3 으로 고정. 더 큰 값(2e-3)에서는 코드가 한 곳으로 몰려 무너지는 경우가 있었다(개발 중 확인한 사실)
''')
b.code('''
# 셀 1 · 설정
import sys, time, json, inspect; sys.path.insert(0, ".")
from wm_vision import *
CFG = dict(pairs=50000, test_pairs=5000, seeds=(0, 1, 2), lam_steps=2000, dyn_steps=1200, tok_steps=2000, tf_steps=1500, tf_seeds=(0,), lr=1e-3, batch=128)
if not IS_GPU:      # GPU 가 없으면 시간 안에 끝나도록 줄인다 — 결과 품질이 낮아질 수 있음
    CFG.update(seeds=(0,), lam_steps=500, dyn_steps=400, tok_steps=600, tf_steps=500)
print("장치 :", DEV, "| 설정 :", CFG)
RES = {"설정": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()} | {"장치": str(DEV)}}
t_all = time.time()
''')

b.md('''
### 셀 2 · 환경과 프레임 쌍
- **왜** : 모델이 받는 재료가 무엇인지 눈으로 본다. 학습 자료는 위 줄(프레임 t)과 아래 줄(프레임 t+1)의 쌍이다
- **읽는 법** : 제목의 「숨은 행동」은 자료를 만든 쪽만 아는 정보이고 학습에는 들어가지 않는다. 마지막 열은 벽에 막힌 경우 — 「우」를 눌렀지만 사각형이 안 움직였다. 이 경우 화면만 보면 「정지」와 구분되지 않는다(셀 5 에서 중요해진다)
- 배경의 희미한 점은 프레임마다 새로 깜빡이는 방해 요소 — 행동과 아무 상관이 없다
''')
b.code('''
# 셀 2 · 환경과 프레임 쌍
t = time.time(); P = make_pairs(CFG["pairs"], seed=0); TE = make_pairs(CFG["test_pairs"], seed=999)
print(f"학습 쌍 {len(P['a']):,}개 · 시험 쌍 {len(TE['a']):,}개 (서로 다른 시드로 만듦) · 생성 {time.time()-t:.1f}초 · 프레임 {P['f0'].shape[1:]} uint8")
pick = [int(np.where(P["a"] == k)[0][0]) for k in range(5)]
pick.append(int(np.where((P["a"] != 0) & (P["pos0"] == P["pos1"]).all(1))[0][0]))        # 벽에 막힌 예
names = [f"숨은 행동: {ACTIONS[P['a'][i]]}" + ("\\n(벽에 막혀 제자리)" if k == 5 else "") for k, i in enumerate(pick)]
show_frames([P["f0"][i] / 255 for i in pick] + [P["f1"][i] / 255 for i in pick], names + ["프레임 t+1"] * 6, ncols=6, size=1.9,
            suptitle="위 : 프레임 t   아래 : 프레임 t+1")
blocked = float(((P["a"] != 0) & (P["pos0"] == P["pos1"]).all(1)).mean())
print("행동 분포(정지·상·하·좌·우) :", np.round(np.bincount(P["a"]) / len(P["a"]), 3), "· 정지가 아닌데 벽에 막힌 쌍 :", f"{blocked*100:.1f}%")
''')

b.md('''
### 셀 3 · VQ(벡터 양자화)를 손으로
- **VQ** : 연속 벡터를 코드북(대표 벡터 K개 목록)에서 **가장 가까운 것 하나로 바꾸고 그 번호를 쓴다**. 연속 값이 K개 중 하나의 번호로 압축된다
- **문제** : 「가장 가까운 것 선택하기」는 미분이 안 된다 → 인코더에 학습 신호가 가지 않는다
- **직통 경사(straight-through)** : 앞으로 갈 때는 치환된 코드 q 를 쓰고, 뒤로 올 때는 치환이 없었던 것처럼 z 로 경사를 그대로 넘긴다. 아래에서 경사가 정말 통과하는지 확인한다
- 손실은 두 항 : 코드북을 z 쪽으로 끌어당김 + z 를 코드북 쪽으로 끌어당김(커밋 항, 가중치 0.25)
''')
b.code('''
# 셀 3 · VQ 를 손으로
cb = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])                 # 코드북 K=4, 차원 2
z = torch.tensor([[0.1, 0.2], [0.9, 0.1], [0.2, 0.8], [0.7, 0.9]], requires_grad=True)
dist = ((z[:, None] - cb[None]) ** 2).sum(-1)                              # 모든 코드와의 거리 제곱 [4 × 4]
idx = dist.argmin(1); q = cb[idx]
print("거리 제곱 :\\n", dist.detach().round(decimals=2))
print("가장 가까운 코드 번호 :", idx.tolist(), "→ 치환된 벡터\\n", q)
q_plain = q                                                                 # 그냥 치환
q_st = z + (q - z).detach()                                                 # 직통 경사 : 값은 q, 경사는 z 로
print("값은 같은가 :", bool(torch.allclose(q_st, q)))
q_st.sum().backward()
print("직통 경사 : z.grad =", z.grad.tolist(), "(전부 1 = 경사가 치환을 건너 인코더까지 도착)")
print("그냥 치환 q 는 계산 그래프에 z 가 없음 : q.requires_grad =", q_plain.requires_grad)
v = vq_quantize(z.detach(), cb)                                             # 우리가 쓸 함수와 같은 결과인지
print("라이브러리 함수와 코드 번호 일치 :", bool((v[1] == idx).all()), "· VQ 손실", round(float(v[2]), 4))
''')

b.md('''
### 셀 4 · 잠재 행동 모델(LAM) 학습 — 행동 라벨 없이
- **구조** : 인코더가 (프레임 t, 프레임 t+1) 을 보고 벡터 하나를 내고 → VQ 가 코드북 K개 중 하나로 치환 → **디코더(= 동역학)** 가 (프레임 t, 코드) 만 받아 프레임 t+1 을 복원한다
- **왜 코드가 변화 정보를 담게 되는가** : 디코더는 프레임 t+1 을 못 본다. 이동 방향을 알려 줄 통로가 코드 하나뿐이어서, 복원 오차를 줄이려면 코드가 「어느 쪽으로 움직였는가」를 담아야 한다. 깜빡이는 점은 코드로 보내 봐야 오차가 거의 안 줄어서 무시된다
- 손실 = 복원 오차(프레임당 제곱합) + VQ 표준 항. 안 쓰이는 코드는 100스텝마다(학습 앞 60% 동안) 현재 인코더 출력으로 다시 심는다(죽은 코드 되살리기)
- **K = 2 · 5 · 8** 을 비교한다. 진짜 행동은 5가지. K 가 모자라면 합쳐지고, 남으면 쪼개지거나 안 쓰인다
- 아래 첫 줄에서 **학습 함수가 받는 인자에 진짜 행동이 없음**을 출력으로 확인한다
''')
b.code('''
# 셀 4 · 잠재 행동 모델 학습
print("학습 함수 인자 :", inspect.signature(train_lam))
D = to_dev(P, ("f0", "f1"))                       # 진짜 행동 a · 위치 pos 는 장치에 올리지도 않는다
assert set(D) == {"f0", "f1"} and "a" not in inspect.signature(train_lam).parameters
print("학습에 들어가는 자료 :", sorted(D), "→ 진짜 행동 없음")
LAMS, rows = {}, []
for K in (2, 5, 8):
    for sd in CFG["seeds"]:
        t = time.time(); m = train_lam(K, D["f0"], D["f1"], CFG["lam_steps"], seed=sd, lr=CFG["lr"], bs=CFG["batch"])
        LAMS[K, sd] = m; rows.append((K, sd, round(m.final_rec, 2), round(time.time() - t, 1)))
R4 = pd.DataFrame(rows, columns=["코드 수 K", "시드", "마지막 100스텝 복원 오차(프레임당 제곱합)", "학습 초"])
print(R4.to_string(index=False))
print(f"설정 : 스텝 {CFG['lam_steps']} · 배치 {CFG['batch']} · 학습률 {CFG['lr']} · 시드 {CFG['seeds']}")
''')

b.md('''
### 셀 5 · 코드의 정체 측정 — 코드는 숨은 진짜 행동과 얼마나 일치하는가
- **왜** : 이제야 숨겨 둔 진짜 행동을 꺼내 **채점**한다(학습에는 쓰지 않았다). 코드가 사람이 정한 버튼이 아니라 학습이 정한 것이라면, 일치가 완벽하지 않을 수 있다
- **지표**
  - **혼동행렬** : 행 = 진짜 행동, 열 = 코드. 한 열에 한 행만 크게 켜지면 그 코드는 그 행동 하나다
  - **순도** : 코드마다 가장 많은 행동의 비율을 가중 평균(1.0 = 코드가 곧 행동)
  - **상호정보량(비트)** : 코드를 알면 진짜 행동에 대해 몇 비트를 아는가. 상한은 min(행동 엔트로피, log₂K). K=2 이면 1비트가 한계다
  - **안 쓰이는 코드 수** : 시험 자료에서 한 번도 선택되지 않은 코드(코드북 붕괴)
- **벽에 막힌 경우를 따로 본다** : 「우」를 눌렀는데 벽이라 안 움직인 쌍은 화면에서 「정지」와 똑같아서, 영상만 보는 모델은 둘을 구분할 수 없다. 그래서 라벨을 두 가지로 채점한다 — 누른 행동(명령) / 실제로 일어난 이동(정지 · 상 · 하 · 좌 · 우)
''')
b.code('''
# 셀 5 · 코드의 정체 측정
TEd = to_dev(TE, ("f0", "f1"))
lab_cmd = TE["a"]                                                                         # 누른 행동
dpos = TE["pos1"] - TE["pos0"]; lab_eff = (dpos[:, None, :] == DELTA[None]).all(2).argmax(1)   # 실제 이동
rows, STAT = [], {}
for (K, sd), m in LAMS.items():
    code = lam_codes(m, TEd["f0"], TEd["f1"])
    s1, s2 = code_stats(code, lab_cmd, K), code_stats(code, lab_eff, K)
    STAT[K, sd] = (code, s1, s2)
    rows.append((K, sd, s1["purity"], s1["mi"], s2["purity"], s2["mi"], s1["dead"], s1["perplexity"]))
R5 = pd.DataFrame(rows, columns=["K", "시드", "순도(누른 행동)", "상호정보(비트·누른 행동)", "순도(실제 이동)", "상호정보(비트·실제 이동)", "안 쓰이는 코드", "코드 퍼플렉시티"])
print("행동 엔트로피 H =", round(code_stats(STAT[2, CFG["seeds"][0]][0], lab_cmd, 2)["H_label"], 3), "비트(최대 log2 5 = 2.322) · 실제 이동 라벨 엔트로피 =", round(code_stats(STAT[2, CFG["seeds"][0]][0], lab_eff, 2)["H_label"], 3), "비트")
print("\\n[시드별]"); print(R5.round(3).to_string(index=False))
S5 = R5.groupby("K").agg(["mean", "std"]).drop(columns="시드").round(3)
print("\\n[K 별 평균, 표준편차(시드 간)]"); print(S5.to_string())
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, K in zip(axs, (2, 5, 8)):
    sd0 = min(CFG["seeds"], key=lambda s: LAMS[K, s].final_rec)          # 복원 오차가 가장 낮은 시드(진짜 행동을 보지 않고 선택)
    C = STAT[K, sd0][1]["C"]; ax.imshow(C / C.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(5)); ax.set_yticklabels(ACTIONS); ax.set_xlabel("코드 번호"); ax.set_title(f"K={K} (복원 오차 최소 시드 {sd0}) — 진짜 행동별 코드 비율", fontsize=9)
    for i in range(5):
        for j in range(K):
            ax.text(j, i, int(C[i, j]), ha="center", va="center", fontsize=7, color="k")
plt.tight_layout(); plt.savefig(RESULTS / "04_confusion.png", dpi=130); plt.show()
RES["코드정체"] = {f"K={K}": {"순도_누른행동_평균": float(g["순도(누른 행동)"].mean()), "상호정보_비트_누른행동_평균": float(g["상호정보(비트·누른 행동)"].mean()),
                              "순도_실제이동_평균": float(g["순도(실제 이동)"].mean()), "상호정보_비트_실제이동_평균": float(g["상호정보(비트·실제 이동)"].mean()),
                              "안쓰이는코드_평균": float(g["안 쓰이는 코드"].mean()), "시드수": int(len(g))} for K, g in R5.groupby("K")}
''')

b.md('''
### 셀 6 · 코드에 이름 붙이기와 플레이
- **왜** : 코드는 번호일 뿐이다. 사람이 쓰려면 「이 코드를 넣으면 화면이 어떻게 바뀌는가」를 **눌러 보고 이름을 붙여야** 한다. 아래는 그 과정을 자동으로 흉내 낸다 — 가운데 사각형 하나를 놓고 코드를 하나씩 넣어서 사각형이 어느 쪽으로 움직이는지 본다. **진짜 행동 라벨은 쓰지 않는다**
- 같은 방향 이름이 붙은 코드가 둘 이상일 수 있고(쪼개짐), 어느 방향도 아닌 코드가 있을 수 있다
- 이어서 이름을 믿고 코드열을 선택해 프레임을 10스텝 굴린다. 마지막에 「이름대로 진짜 환경을 움직인 경로」와 비교해 이름이 맞았는지 확인한다
- 선택한 시드 : K=8 모델 중 **복원 오차가 가장 낮은 시드**(진짜 행동을 보지 않고 선택하는 기준)
''')
b.code('''
# 셀 6 · 코드에 이름 붙이기와 플레이
sd8 = min(CFG["seeds"], key=lambda s: LAMS[8, s].final_rec); m8 = LAMS[8, sd8]
print("선택한 시드 :", sd8, "(K=8 복원 오차 최소)")
probe = torch.tensor(render(np.array([[14, 14]]), np.array([0]), np.random.default_rng(0), dots=0), device=DEV)
use = np.bincount(STAT[8, sd8][0], minlength=8) / len(STAT[8, sd8][0])
NAME2ACT = {"정지": 0, "위": 1, "아래": 2, "왼쪽": 3, "오른쪽": 4}
code_name, rows = {}, []
with torch.no_grad():
    for c in range(8):
        nxt = m8.dyn(probe, m8.codebook.weight[c][None]).clamp(0, 1)
        d = (centroid(nxt)[0] - centroid(probe)[0]).cpu().numpy()
        if np.isnan(d).any(): nm = "불명"
        elif np.abs(d).max() < 0.7: nm = "정지"
        elif abs(d[0]) >= abs(d[1]): nm = "오른쪽" if d[0] > 0 else "왼쪽"
        else: nm = "아래" if d[1] > 0 else "위"
        code_name[c] = nm; rows.append((c, nm, np.round(d, 2).tolist(), round(float(use[c]) * 100, 1)))
print(pd.DataFrame(rows, columns=["코드", "붙인 이름", "사각형 중심 이동(dx, dy)", "시험 자료에서 쓰인 비율 %"]).to_string(index=False))
MY_CODES = None        # 직접 선택하려면 코드 번호 10개 리스트를 넣는다. 예 [3, 3, 3, 5, 5, 5, 0, 0, 1, 1]
if MY_CODES is None:
    plan = [("오른쪽", 4), ("아래", 3), ("왼쪽", 2), ("위", 1)]
    best = {nm: max([c for c in range(8) if code_name[c] == nm], key=lambda c: use[c], default=None) for nm in NAME2ACT}
    MY_CODES = [best[nm] for nm, n in plan for _ in range(n) if best[nm] is not None]
    print("방향별로 가장 많이 쓰이는 코드 :", best)
print("롤아웃에 넣은 코드열 :", MY_CODES)
x0 = render(np.array([[4, 4]]), np.array([1]), np.random.default_rng(7))
frames = lam_rollout(m8, torch.tensor(x0).mul(255).round().to(torch.uint8).to(DEV), MY_CODES)
show_frames(frames, ["시작"] + [f"{k+1}: 코드{c}\\n{code_name[c]}" for k, c in enumerate(MY_CODES)], ncols=11, size=1.25, suptitle="코드열을 넣어 굴린 프레임(처음 한 장만 진짜, 나머지는 모델이 만든 것)")
pred = np.array([centroid(torch.tensor(f[None], device=DEV))[0].cpu().numpy() for f in frames])
pos = np.array([[4, 4]]); path = [pos[0] + HALF]
for c in MY_CODES:
    pos = step_pos(pos, np.array([NAME2ACT[code_name[c]]])); path.append(pos[0] + HALF)
path = np.array(path)
print("모델이 만든 사각형 중심 :", np.round(pred, 1).tolist())
print("이름대로 진짜 환경을 움직인 중심 :", path.tolist())
print("두 경로의 최대 차이(픽셀) :", round(float(np.nanmax(np.linalg.norm(pred - path, axis=1))), 2))
''')

b.md('''
### 셀 7 · 행동 조건화 세 방식 비교 — 행동을 어떻게 넣는가
- **왜** : 앞 셀들은 코드를 쓰지만, 진짜 행동 5가지를 입력으로 줄 수 있다고 가정하면 그 값을 **모델에 어떻게 넣는가**가 성능을 가른다. 같은 작은 CNN 에서 넣는 방식만 바꾼다
  - ⓐ **무시** : 행동을 안 넣는다(대조군)
  - ⓑ **concat** : 행동 임베딩을 입력 채널에 이어 붙인다
  - ⓒ **FiLM / AdaLN** : 층마다 정규화를 한 뒤 `(1+γ)·h + β` 로 **행동이 곱과 더하기 값을 조절**한다. 행동이 모든 층에 걸쳐 특징을 직접 조절하는 방식
- **지표**
  - **1스텝 MSE** : 진짜 행동을 넣었을 때 다음 프레임의 픽셀당 오차. 같은 표에 「프레임 t 를 그대로 복사」한 값을 대조로 둔다
  - **행동 민감도(정합률)** : 같은 프레임에서 행동만 바꿔 예측했을 때, 행동 a 와 b 의 예측 사각형 중심 차이가 진짜 중심 차이와 1픽셀 이내로 맞는 비율(진짜 결과가 다른 쌍만 센다). 행동에 반응하지 못하는 모델은 0 에 가깝다
  - **10스텝 개루프 오차** : 처음 한 장만 진짜이고 이후는 모델 출력을 다시 입력으로 쓰며 무작위 행동 10번. 사각형 중심의 위치 오차(픽셀). 사각형을 잃어버린 비율도 같이 본다
''')
b.code('''
# 셀 7 · 행동 조건화 세 방식 비교
D2 = to_dev(P, ("f0", "f1", "a")); T2 = to_dev(TE, ("f0", "f1", "a"))
NS, NE, H = 2000, 1000, 10
ct = true_centers_all_actions(TE["pos0"][:NS])
acts_np = np.random.default_rng(5).integers(0, 5, (NE, H)); acts = torch.tensor(acts_np, device=DEV)
tc = true_centers_after(TE["pos0"][:NE], acts_np)
copy_mse = float(((fl(T2["f0"]) - fl(T2["f1"])) ** 2).mean())
rows, curves = [], {}
for cond in ("none", "concat", "film"):
    for sd in CFG["seeds"]:
        t = time.time(); m = train_dyn(Dyn(cond), D2, CFG["dyn_steps"], bs=CFG["batch"], lr=CFG["lr"] * 2, seed=sd)
        cp = centers_all_actions(m, T2["f0"][:NS])
        err, lost = pos_error(rollout_centers(m, T2["f0"][:NE], acts), tc)
        curves.setdefault(cond, []).append(err)
        rows.append((cond, sd, mse_one_step(m, T2), pair_sensitivity(cp, ct), err[-1], lost[-1], time.time() - t))
R7 = pd.DataFrame(rows, columns=["방식", "시드", "1스텝 MSE", "행동 민감도(정합률)", "10스텝 위치오차(px)", "10스텝 사각형 잃은 비율", "학습 초"])
lab = {"none": "ⓐ 무시", "concat": "ⓑ concat", "film": "ⓒ FiLM"}
print(f"설정 : 스텝 {CFG['dyn_steps']} · 배치 {CFG['batch']} · 학습률 {CFG['lr']*2} · 시드 {CFG['seeds']} · 시험 프레임 {NS}(민감도) / {NE}(롤아웃)")
print(f"참고 : 프레임 t 를 그대로 복사했을 때 1스텝 MSE = {copy_mse:.5f}")
print("\\n[시드별]"); print(R7.round(5).to_string(index=False))
T7 = pd.DataFrame({lab[c]: {k: ms(g[k].values, 5 if k == "1스텝 MSE" else 3) for k in R7.columns[2:6]} for c, g in R7.groupby("방식", sort=False)})
print("\\n[평균±표준편차(시드 간)]"); print(T7.to_string())
print("(nan = 롤아웃 10스텝에서 사각형을 전부 잃어 위치 오차를 구할 수 없음)")
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for c in ("none", "concat", "film"):
    arr = np.array(curves[c]); ax.errorbar(range(1, H + 1), np.nanmean(arr, 0), yerr=np.nanstd(arr, 0), marker="o", label=lab[c], capsize=2)
ax.set_xlabel("롤아웃 스텝"); ax.set_ylabel("사각형 위치 오차(px)"); ax.set_title("개루프 오차 — 스텝이 늘수록 쌓이는가"); ax.legend(); plt.tight_layout(); plt.show()
RES["조건화"] = {"1스텝_MSE_복사기준": copy_mse} | {lab[c]: {"1스텝_MSE_평균": float(g["1스텝 MSE"].mean()), "행동민감도_정합률_평균": float(g["행동 민감도(정합률)"].mean()),
                    "개루프10스텝_위치오차px_평균": float(g["10스텝 위치오차(px)"].mean()), "시드수": int(len(g))} for c, g in R7.groupby("방식", sort=False)}
''')

b.md('''
### 셀 8 · 프레임을 4×4 토큰으로 — 작은 VQ 토크나이저
- **왜** : 반복 마스크 채우기는 **토큰**을 채우는 기법이다. 먼저 프레임(32×32×3)을 토큰 격자(4×4 = 토큰 16개, 토큰 하나가 8×8 픽셀 조각, 코드북 128개)로 바꾸고 다시 그림으로 되돌리는 VQ 토크나이저를 만든다
- **읽는 법** : 복원이 완벽하지 않다 — 이 복원 품질이 뒤 셀 모든 결과의 **상한**이다. 토큰이 잘못되면 어떤 디코딩도 이 상한 이상으로 좋아질 수 없다
- 토크나이저는 학습 쌍의 프레임 t · t+1 일부로 학습한다(행동 라벨 불필요)
- 코드북이 작아서 색이 틀리게 복원되거나 사각형이 흐려지는 프레임이 있다. 결과 그림에서 직접 확인한다
''')
b.code('''
# 셀 8 · 프레임을 4×4 토큰으로
nh = CFG["pairs"] // 2
t = time.time(); tk = train_tok(torch.cat([D["f0"][:nh], D["f1"][:nh]]), CFG["tok_steps"], seed=0, lr=CFG["lr"], bs=CFG["batch"])
print(f"토크나이저 학습 {time.time()-t:.0f}초 · 스텝 {CFG['tok_steps']} · 학습률 {CFG['lr']} · 코드북 {tk.K}개 · 토큰 격자 4×4")
xt = fl(TEd["f1"][:2000])
with torch.no_grad(): rec_img = tk(xt)[0].clamp(0, 1)
tok_mse = float(((rec_img - xt) ** 2).mean()); tok_coh = coherent_rate(rec_img)
print(f"시험 프레임 복원 MSE {tok_mse:.5f} · 복원해도 사각형이 온전한 비율 {tok_coh:.3f} (원본은 {coherent_rate(xt):.3f})")
tt = tk.encode(xt[:3])
print("토큰 격자 예(첫 프레임) :\\n", tt[0].reshape(4, 4).cpu().numpy())
show_frames([xt[i].cpu().numpy() for i in range(6)] + [rec_img[i].cpu().numpy() for i in range(6)], ["원본"] * 6 + ["토큰으로 복원"] * 6, ncols=6, size=1.9)
''')

b.md('''
### 셀 9 · 반복 마스크 채우기 — 한 번에 vs 여러 번에 나눠
- **하는 일** : 이전 프레임의 토큰 16개(보임)를 받아 **다음 프레임의 토큰 16개**를 만든다. 작은 양방향 트랜스포머를 「토큰 일부를 가려 두고 맞히기」로 학습한다
- **디코딩** : 전부 [MASK] 에서 시작해, 매번 모든 칸을 예측하되 **가려진 칸 중 확신(최대 확률)이 높은 칸부터** 확정하고 나머지는 다시 가린다. R 번에 나눠 채운다. **R=1 은 한 번에 전부 예측**
- **두 조건** : (A) 행동을 알려 준다 → 다음 프레임이 거의 하나로 정해진다 (B) 행동을 숨긴다 → 다음 프레임이 여러 가지로 갈린다(5가지 이동)
- **지표**(다음 프레임 정답과 비교)
  - 토큰 정답률 : 정답 프레임을 토크나이저에 넣어 얻은 토큰과 같은 칸의 비율
  - 픽셀 MSE · 사각형 중심 위치 오차(px) · 사각형을 잃은 비율
  - **온전한 사각형 비율** : 복원 그림에 사각형 하나가 온전히 있는가. 칸마다 따로 예측하면 서로 어긋난 조각이 나올 수 있고, 이 지표가 그것을 잡는다
- **정직하게** : 어느 쪽이 이길지 미리 정하지 않는다. 값이 R 에 따라 단조롭게 나아지지 않으면 그대로 본다. 복원 상한은 셀 8 의 값
''')
b.code('''
# 셀 9 · 반복 마스크 채우기
def tok_all(u8): return torch.cat([tk.encode(fl(u8[i:i + 1000])) for i in range(0, len(u8), 1000)])
ptr0, ptr1, tte0, tte1 = tok_all(D2["f0"]), tok_all(D2["f1"]), tok_all(T2["f0"]), tok_all(T2["f1"])
print(f"학습 토큰 {tuple(ptr0.shape)} · 쓰인 코드 {len(ptr1.unique())}/{tk.K}")
tcen, f1_te = true_center(TE["pos1"]), fl(T2["f1"])
Rs, rows, ex = (1, 2, 4, 8), [], {}
for use_act in (True, False):
    for sd in CFG["tf_seeds"]:
        t = time.time(); tf = train_mask_tf(MaskTf(tk.K, use_act=use_act), ptr0, ptr1, D2["a"], CFG["tf_steps"], seed=sd)
        for R in Rs:
            tok = mask_decode(tf, tte0, T2["a"], R); fr = tk.decode(tok).clamp(0, 1)
            cp = centroid(fr); lost = torch.isnan(cp).any(1)
            rows.append(("A 행동 있음" if use_act else "B 행동 숨김", sd, R, float((tok == tte1).float().mean()), float(((fr - f1_te) ** 2).mean()),
                         float(torch.nanmean((cp - tcen).norm(dim=1))), float(lost.float().mean()), coherent_rate(fr)))
            if sd == CFG["tf_seeds"][0]: ex[use_act, R] = fr[:6].cpu().numpy()
        print(f"트랜스포머 학습 {'A' if use_act else 'B'} 시드 {sd} : {time.time()-t:.0f}초")
R9 = pd.DataFrame(rows, columns=["조건", "시드", "R(채우는 횟수)", "토큰 정답률", "픽셀 MSE", "중심 위치 오차(px)", "사각형 잃은 비율", "온전한 사각형 비율"])
print(f"설정 : 스텝 {CFG['tf_steps']} · 시드 {CFG['tf_seeds']} · 층 4 · 폭 128 · 확신 높은 칸부터 확정(argmax) · 시험 {len(tte0)}쌍 · 토크나이저 복원 상한: MSE {tok_mse:.5f}, 온전한 사각형 비율 {tok_coh:.3f}")
print(R9.groupby(["조건", "R(채우는 횟수)"]).mean().drop(columns="시드").round(4).to_string())
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
for cnd, g in R9.groupby("조건"):
    g = g.drop(columns="조건").groupby("R(채우는 횟수)").mean()
    axs[0].plot(g.index, g["픽셀 MSE"], "o-", label=cnd); axs[1].plot(g.index, g["온전한 사각형 비율"], "o-", label=cnd)
axs[0].set_title("픽셀 MSE (낮을수록 좋음)"); axs[1].set_title("온전한 사각형 비율 (높을수록 좋음)")
for a_ in axs: a_.set_xscale("log", base=2); a_.set_xticks(Rs); a_.set_xticklabels(Rs); a_.set_xlabel("R"); a_.legend()
plt.tight_layout(); plt.show()
show_frames([f1_te[i].cpu().numpy() for i in range(6)] + [ex[False, 1][i] for i in range(6)] + [ex[False, 8][i] for i in range(6)],
            ["정답"] * 6 + ["B · R=1(한 번에)"] * 6 + ["B · R=8(나눠서)"] * 6, ncols=6, size=1.9, suptitle="행동을 숨긴 조건 — 같은 입력의 정답과 두 디코딩")
RES["마스크채우기"] = {"토크나이저_복원MSE": tok_mse, "토크나이저_온전한사각형비율": tok_coh} | {f"{k[0]}_R={k[1]}": {c: float(v) for c, v in r.items()} for k, r in
                       R9.groupby(["조건", "R(채우는 횟수)"]).mean().drop(columns="시드").iterrows()}
''')

b.md('''
### 셀 10 · 정리 · 결과 저장
- 위 표의 숫자가 이 실습의 근거다. 이 장난감 세계의 결과이고, 시드 · 스텝을 바꾸면 달라질 수 있다(특히 코드가 어떤 행동에 대응하는지)
''')
b.code('''
# 셀 10 · 정리와 결과 저장
RES["총_실행_초"] = round(time.time() - t_all, 1)
(RESULTS / "measured_04.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
print("저장 :", RESULTS / "measured_04.json", "· 전체 실행", RES["총_실행_초"], "초")
print(json.dumps(RES["코드정체"], ensure_ascii=False, indent=1))
''')

b.md('''
## 정리

| 본 것 | 실험에서 확인한 방법 |
|---|---|
| 라벨 없이 변화 코드가 생긴다 | 셀 4 : 인자에 행동이 없는 학습 → 셀 5 에서 비로소 채점 |
| 코드는 학습이 정한 것 | 셀 5 : 순도 · 상호정보량이 1 이 아니고 시드마다 다름. 막힌 이동은 구분 불가 |
| 이름은 사람이 붙인다 | 셀 6 : 코드를 눌러 보고 이름 붙임 → 이름대로 진짜 환경을 움직인 경로와 비교 |
| 행동을 넣는 방식이 성능을 가른다 | 셀 7 : 행동 민감도 · 개루프 오차 (1스텝 MSE 만으로는 안 보임) |
| 여러 번에 나눠 채우는 디코딩 | 셀 9 : R 별 비교 — 행동이 있을 때와 숨길 때 |

### 확인 질문
1. 셀 5 에서 K=2 의 상호정보량 상한은 왜 1비트인가? K 를 5 로 늘리면 상호정보량이 늘어나는 이유와, 그래도 2.32비트(행동 엔트로피)에 못 미치는 이유를 각각 설명해 보자.
2. 벽에 막힌 「우」와 「정지」를 영상만 보는 모델이 구분하지 못하는 것은 모델의 결함인가, 자료의 성질인가? 이 성질은 실제 게임 영상에서 어떤 형태로 나타날까?
3. 셀 7 에서 행동 무시 모델의 1스텝 MSE 는 「프레임 그대로 복사」 기준선과 비교하면 어떤가? 그 숫자만 봤다면 이 모델을 걸러낼 수 있었을까, 행동 민감도와 10스텝 롤아웃에서는 무엇이 드러났는가?
''')

b.save()
