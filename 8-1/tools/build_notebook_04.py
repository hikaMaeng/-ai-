"""실습 노트북 04 생성기 (강사용).

    python build_notebook_04.py   → ../notebooks/04_복리_오류와_모델_크기.ipynb (출력 없이)
"""
from nb import Book

b = Book("04_복리_오류와_모델_크기.ipynb")

b.md('''
# 실습 04 · 복리 오류와 모델 크기 — 스텝이 길어질수록 성공률은 곱해져서 내려간다

**직접 해 보는 것** : 스텝마다 성공할 확률이 p 라면 n 스텝짜리 궤적이 끝까지 성공할 확률은 `p^n` 이다. 이 계산이 맞는 조건을 가짜 정책 시뮬레이션으로 먼저 확인하고(서버 없이 계산됨), 같은 구조의 연쇄 조회 과제를 진짜 모델에게 길이별로 풀게 해 실제 성공률이 이론 곡선과 얼마나 닿는지 본다. 마지막으로 모델 크기를 바꿔 가며 도구 호출 · 지시준수 · 긴 연쇄 세 가지가 어디서부터 성립하는지 본다.

**강의의 어느 개념과 이어지는가**
- 도구 호출 · 지시준수 · 추론 능력이 임계를 넘어야 자율 루프가 성립한다(능력 임계)
- 자율성이 커질수록 스텝별 오류가 궤적으로 누적 · 증폭된다 = **복리 오류**
- 스텝 성공률이 0.95 로 높아도 20 스텝이면 `0.95^20 ≈ 0.36` — 길이가 곱해져 급락한다
- 긴 궤적일수록 **중간 검증**과 **복구 지점**이 필요하다

**문제** : (1) 이론 곡선과 가짜 정책 시뮬레이션이 맞는가, 복구가 있으면 곡선이 어떻게 달라지는가. (2) 노트북이 만든 합성 사슬(`K0` → `get` 으로 다음 키 → … n 번 따라간 마지막 값)에서 길이별 실제 성공률은 이론과 맞는가. (3) 모델 크기에 따라 유효한 도구 호출 비율 · 첫 행동 정확도 · 길이 8 연쇄 성공률은 어떻게 달라지는가. 정답은 코드가 알고 있다(사람이 채점하지 않음).

| 셀 | 하는 일 |
|---|---|
| 1 | 설정 · LLM 호출 횟수 계산 · 모델 목록 |
| 2 | 이론 — `p^n` 곡선과 손 계산 |
| 3 | 가짜 정책 시뮬레이션 — 독립 오류 · 복구 없음 / 복구 있음 / 오류가 안 보이는 경우 (서버 필요 없음) |
| 4 | 연쇄 조회 과제 만들기 · 진짜 모델로 길이별 실행 |
| 5 | 실측 곡선 vs 이론 |
| 6 | 모델 크기 비교 ① — 지금 서버에 로드된 모델을 측정해 캐시에 쌓기 |
| 7 | 모델 크기 비교 ② — 캐시가 쌓인 모델만 모아 표 · 그림 |
| 8 | 정리 · 결과 저장 · 확인 질문 |

- 실행 : 셀 2 · 3 은 서버 없이 계산된다. 셀 4 · 6 은 LM Studio 서버가 켜져 있고 도구 호출이 되는 모델이 로드되어 있어야 한다(없으면 `AGENT_MODE=replay` 로 강사 실행 결과를 재생). 시드 · 표본 수 · 온도 같이 결과를 바꾸는 설정은 모두 출력에 찍는다
- **결과가 기대와 달라도 그대로 남긴다.** 실측이 이론과 어긋나면 어긋난 숫자를 그대로 보고 이유를 따져 본다
- 모델은 한 번에 하나만 서버에 로드된다. 모델 비교는 「모델을 바꿔 로드 → 셀 6 을 다시 실행」으로 모델마다 캐시를 쌓는 방식이다
''')

b.md('''
### 셀 1 · 설정
- `lengths` : 사슬 길이(조회 횟수). `n_chains` : 길이마다 서로 다른 사슬 수. `sim_trials` : 시뮬레이션 반복 수
- 아래에서 이 노트북이 LLM 을 부르는 횟수를 계산해 찍는다. 캐시에 있는 호출은 서버를 부르지 않는다
''')
b.code('''
# 셀 1 · 설정
import sys, json, time, math; sys.path.insert(0, ".")
from agent_lab import *
from agent_tasks import *

CFG = dict(lengths=[1, 2, 4, 8, 12], n_chains=8, max_tokens=400, extra_steps=3, temperature=0.0, seed=0,
           sim_trials=1000, sim_lengths=[1, 2, 4, 8, 12, 20, 30], sim_p=[0.99, 0.95, 0.90], sim_p_focus=0.95, sim_seed=7, recover_retries=2)
N_CMP = 8                                                    # 모델 비교에서 쓰는 사슬 길이
MODELS = ["qwen3.5-0.8b-mtp", "qwen3.5-2b-mtp", "qwen3.5-4b-mtp", "qwen3.5-9b-mtp"]
llm = LLM()          # 주소 · 모델은 환경변수(LMSTUDIO_URL · AGENT_MODEL)에서 읽는다
orders, cust, QS = make_questions_D(0)
QD = [q for q in QS if q["유형"] in ("주문번호", "이름+상품")]     # 첫 행동 정확도를 보는 질문(정답 첫 행동 : 주문번호 → lookup_order, 이름+상품 → search_customer)
TOOLS_D = tools_D(orders, cust)

chain_calls = CFG["n_chains"] * sum(n + 1 for n in CFG["lengths"])        # 길이 n 이면 조회 n 번 + 최종 답 1번
print(llm, "| 설정 :", CFG)
print("비교할 모델 :", MODELS)
print(f"이 노트북이 LLM 을 부르는 횟수(모델 하나당) : 약 {chain_calls + len(QD)}번 — 연쇄 조회 약 {chain_calls}번 + 첫 행동 시험 {len(QD)}번. 캐시에 있으면 서버를 부르지 않는다")
print(f"  다른 모델로 바꿔 셀 6 만 다시 돌리면 그 모델은 약 {len(QD) + CFG['n_chains'] * (N_CMP + 1)}번 (첫 행동 {len(QD)}번 + 길이 {N_CMP} 사슬 {CFG['n_chains']}개). 셀 2 · 3 의 시뮬레이션은 LLM 을 부르지 않는다")
''')

b.md('''
### 셀 2 · 이론 — 곱해지는 확률
- 스텝마다 성공할 확률이 p 이고, 스텝끼리 **독립**이며, 한 번 틀리면 **복구하지 않는다**고 하면 n 스텝 모두 성공할 확률은 `p^n` 이다
- 아래 그림은 p = 0.99 · 0.95 · 0.90 에서 길이 1~30 의 곡선이다. 먼저 `0.95^20` 을 손으로 곱해 확인한다
''')
b.code('''
# 셀 2 · 이론 곡선과 손 계산
v = 1.0
for i in range(20):
    v *= 0.95
print(f"0.95 를 20번 곱하면 {v:.4f}   (0.95**20 = {0.95 ** 20:.4f})")
print()
N = np.arange(1, 31)
for p in CFG["sim_p"]:
    half = int(math.floor(math.log(0.5) / math.log(p))) + 1          # p^n 이 처음 0.5 아래로 내려가는 길이
    print(f"p = {p:.2f} : n=10 → {p ** 10:.3f} · n=20 → {p ** 20:.3f} · n=30 → {p ** 30:.3f} · 처음 0.5 아래로 내려가는 길이 n = {half}")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for p, c in zip(CFG["sim_p"], ["#3E7A3A", "#D9822B", "#B91C1C"]):
    ax.plot(N, p ** N, color=c, lw=2, label=f"p = {p:.2f}")
ax.scatter([20], [0.95 ** 20], color="#D9822B", zorder=3)
ax.annotate(f"0.95^20 = {0.95 ** 20:.2f}", (20, 0.95 ** 20), xytext=(21.5, 0.5), arrowprops=dict(arrowstyle="-", color="#555"), fontsize=9)
ax.axhline(0.5, color="#bbb", ls=":", lw=1)
ax.set_xlabel("궤적 길이 n (스텝 수)")
ax.set_ylabel("끝까지 성공할 확률 p^n")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False)
ax.set_title("스텝 성공률이 높아도 길이가 곱해져 내려간다", fontsize=11, loc="left")
plt.tight_layout(); plt.show()
''')

b.md('''
### 셀 3 · 가짜 정책으로 시뮬레이션 — 이론이 맞는 조건과 달라지는 조건
- `ScriptedPolicy(eps = 1 − p)` 는 기준 정책이 낸 행동을 확률 eps 로 틀린 행동으로 바꾼다. 서버 없이 도는 진짜 계산이다. 사슬 과제(`get` 으로 키를 따라간다)를 길이별로 각 `sim_trials` 번 돌린다
- 세 가지 조건을 비교한다
  - **복구 없음** : 틀린 호출은 오류 관찰로 돌아오고 정책은 그 뒤로도 정해진 횟수만큼만 부른다 → 독립 오류 · 복구 없음, `p^n` 이 맞아야 하는 조건
  - **복구 있음** : 틀린 호출이 오류 관찰로 보이면 같은 단계를 다시 시도한다. 틀린 호출을 다시 부를 여유는 `recover_retries`(기본 2번) → 맞는 이론은 「조회에 쓸 수 있는 호출 n + 2 번 안에서 성공이 n 번 이상」(이항분포)
  - **오류가 안 보임** : 틀린 호출이 그럴듯한 값을 돌려줘서 틀린 줄 모르고 이어 간다 → 검증 신호가 없으면 복구할 수 없다
- 같은 시드를 쓰면 길이가 달라도 같은 난수열이 겹쳐 칸끼리 오차가 같은 쪽으로 쏠리므로 조건마다 시드를 달리했다
- 표에는 이론 · 시뮬레이션 · 표준오차 `sqrt(p(1−p)/반복 수)` · 차이가 표준오차의 몇 배인지를 찍는다. 3배를 넘는 칸이 있으면 그대로 드러난다
''')
b.code('''
# 셀 3 · 가짜 정책 시뮬레이션
SYSTEM_CHAIN = "너는 조회 도구 get 으로 값을 따라가는 도우미다. 도구 결과는 사실이다. 추측하지 않고, 지시한 횟수만큼 조회한다."

def make_chain(n, seed, n_distract=6):
    """합성 사슬 — K0 → a1 → … → an. 표에는 사슬 밖의 키도 몇 개 들어 있다. 정답은 an"""
    rng = random.Random(seed)
    pool = [f"K{i}" for i in range(1, 100)]
    rng.shuffle(pool)
    path = ["K0"] + pool[:n]
    table = {path[i]: path[i + 1] for i in range(n)}
    extra = pool[n:n + n_distract]
    for k in extra:
        table[k] = rng.choice(pool[n + n_distract:])
    return {"n": n, "path": path, "table": table, "answer": path[n]}

def chain_tools(ch):
    def get(key):
        k = str(key).strip()
        return f"값 : {ch['table'][k]}" if k in ch["table"] else f"오류: 없는 키 {k}"
    return [Tool("get", "키로 값을 조회한다. 값은 다음 키 하나다.", {"properties": {"key": {"type": "string", "description": "예 : K0"}}, "required": ["key"]}, get)]

def chain_user(n):
    return f"키 K0 에서 시작한다. get 도구로 K0 의 값을 조회하고, 그 값을 다시 get 의 키로 넣어 이어서 조회한다. 이렇게 모두 {n}번 조회한 뒤, {n}번째 조회에서 얻은 값만 답한다."

KEYTOK = re.compile(r"(?<![0-9A-Za-z])K[0-9]+(?![0-9])")
def score_chain(ch, text):
    """답에 나온 마지막 키가 정답이면 성공(규칙 채점)"""
    toks = KEYTOK.findall(text or "")
    return bool(toks) and toks[-1] == ch["answer"]

def sim_script(mode, n):
    def script(messages, tools, rng):
        obs = [m["content"] for m in messages if m["role"] == "tool"]
        good = [o for o in obs if o.startswith("값 :")]
        if mode == "복구":                                  # 성공한 마지막 값에서 이어 가고, 실패하면 같은 단계를 다시 부른다
            if len(good) >= n:
                return ("final", good[-1].split(": ")[1])
            return ("call", "get", {"key": good[-1].split(": ")[1] if good else "K0"})
        if len(obs) >= n:                                   # 한 번씩만 부른다
            return ("final", obs[-1].split(": ")[1] if obs[-1].startswith("값 :") else "알 수 없음")
        if not obs:
            return ("call", "get", {"key": "K0"})
        return ("call", "get", {"key": obs[-1].split(": ")[1] if obs[-1].startswith("값 :") else "?"})
    return script

def sim_budget(mode, n):
    return n + 1 + CFG["recover_retries"] if mode == "복구 있음" else n + 2      # 복구 : 조회 n번 + 틀린 호출을 다시 부를 여유 + 최종 답 1번

def sim_rate(mode, p, n, trials, seed):
    ok = 0
    for t in range(trials):
        ch = make_chain(n, 10_000 + t)
        hidden = [k for k in ch["table"] if k not in ch["path"]]
        if mode == "오류가 안 보임":
            corrupt = lambda m, tl, r, hidden=hidden: ("call", "get", {"key": r.choice(hidden)})       # 사슬 밖 키 → 그럴듯한 값이 돌아옴
        else:
            corrupt = lambda m, tl, r: ("call", "get", {"key": "K" + str(r.randint(100, 999))})       # 표에 없는 키 → 오류 관찰
        pol = ScriptedPolicy(sim_script("복구" if mode == "복구 있음" else "한 번씩", n), eps=1 - p, corrupt=corrupt, seed=seed * 1_000_003 + t)
        tr = run_agent(pol, SYSTEM_CHAIN, chain_user(n), chain_tools(ch), max_steps=sim_budget(mode, n))
        ok += score_chain(ch, tr.final)
    return ok / trials

def theory(mode, p, n):
    if mode != "복구 있음":
        return p ** n
    m = sim_budget(mode, n) - 1                             # 마지막 스텝은 최종 답에 쓰인다 → 조회에 쓸 수 있는 호출 수
    return sum(math.comb(m, k) * p ** k * (1 - p) ** (m - k) for k in range(n, m + 1))

t0 = time.time()
rows = []
for mode in ["복구 없음", "복구 있음", "오류가 안 보임"]:
    for p in (CFG["sim_p"] if mode == "복구 없음" else [CFG["sim_p_focus"]]):
        for n in CFG["sim_lengths"]:
            r = sim_rate(mode, p, n, CFG["sim_trials"], CFG["sim_seed"] + 100 * ["복구 없음", "복구 있음", "오류가 안 보임"].index(mode))      # 조건마다 시드를 달리한다
            th = theory(mode, p, n)
            se = math.sqrt(max(th * (1 - th), 1e-12) / CFG["sim_trials"])
            rows.append({"조건": mode, "p": p, "n": n, "이론": th, "시뮬레이션": r, "표준오차": se, "차이/표준오차": (r - th) / se})
SIM = pd.DataFrame(rows)
print(f"시뮬레이션 {len(SIM)}칸 × {CFG['sim_trials']}번, 시드 {CFG['sim_seed']}, 걸린 시간 {time.time() - t0:.0f}초")
print(SIM.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
inside = (SIM["차이/표준오차"].abs() <= 3).sum()
print(f"\\n차이가 표준오차의 3배 이내인 칸 : {inside} / {len(SIM)}   (가장 큰 값 {SIM['차이/표준오차'].abs().max():.2f}배)")
off = SIM[SIM["차이/표준오차"].abs() > 3]
if len(off):
    print("3배를 넘은 칸 :"); print(off.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
Nn = np.arange(1, 31)
for p, c in zip(CFG["sim_p"], ["#3E7A3A", "#D9822B", "#B91C1C"]):
    a1.plot(Nn, p ** Nn, color=c, lw=1.5, alpha=0.7)
    s = SIM[(SIM["조건"] == "복구 없음") & (SIM["p"] == p)]
    a1.errorbar(s["n"], s["시뮬레이션"], yerr=2 * s["표준오차"], fmt="o", color=c, capsize=3, label=f"p = {p:.2f}")
a1.set_title("복구 없음 · 독립 오류 : 선 = 이론 p^n, 점 = 시뮬레이션(막대 ±2 표준오차)", fontsize=10, loc="left")
a1.set_xlabel("궤적 길이 n"); a1.set_ylabel("끝까지 성공한 비율"); a1.set_ylim(0, 1.05); a1.legend(frameon=False)
pf = CFG["sim_p_focus"]
a2.plot(Nn, pf ** Nn, color="#444", lw=1.5, label="이론 p^n (복구 없음)")
a2.plot(Nn, [theory("복구 있음", pf, int(k)) for k in Nn], color="#3E7A3A", lw=1.5, ls="--", label="이론(다시 부를 여유 2번 안에서 복구)")
for mode, c, mk in [("복구 없음", "#444", "o"), ("복구 있음", "#3E7A3A", "s"), ("오류가 안 보임", "#B91C1C", "^")]:
    s = SIM[(SIM["조건"] == mode) & (SIM["p"] == pf)]
    a2.scatter(s["n"], s["시뮬레이션"], color=c, marker=mk, zorder=3, label=f"시뮬레이션 · {mode}")
a2.set_title(f"p = {pf:.2f} 에서 조건별 비교", fontsize=10, loc="left")
a2.set_xlabel("궤적 길이 n"); a2.set_ylim(0, 1.05); a2.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
''')

b.md('''
### 셀 4 · 연쇄 조회 과제를 만들어 진짜 모델에게 풀게 한다
- 사슬 : 키 `K0` → `get(key)` 가 다음 키를 돌려주고, 지시한 횟수 n 만큼 따라간 마지막 값이 정답이다. 도구는 `get` 하나뿐이고 정답은 코드가 안다
- 길이 n ∈ `lengths` 마다 서로 다른 사슬 `n_chains` 개. 사슬은 시드로 정해져 있어 다시 돌려도 같다. 온도 0 · 시드 고정
- 채점 규칙 : 답에 나온 **마지막 키**가 정답이면 성공. 이 셀은 도구 호출 · 지시준수(횟수를 지킴) · 값 이어 가기를 한꺼번에 시험한다
- 한 사슬마다 기록하는 것 : 성공 여부 · 스텝 수 · 조회 횟수 · 오류 관찰 수 · **첫 이탈 위치**(모델이 부른 키가 처음으로 정답 경로에서 벗어난 번호)
''')
b.code('''
# 셀 4 · 연쇄 조회 과제 · 길이별 실행
CHAINS = {n: [make_chain(n, 1000 * n + j) for j in range(CFG["n_chains"])] for n in sorted(set(CFG["lengths"]) | {N_CMP})}
ex = CHAINS[4][0]
print("사슬 예(길이 4) :", " → ".join(ex["path"]), "| 정답", ex["answer"], "| 표에 든 키 수", len(ex["table"]))
print("지시문 :", chain_user(4))

def first_dev(ch, calls):
    """모델이 부른 키가 정답 경로(K0, a1, a2, …)에서 처음 벗어난 번호(1부터). 벗어나지 않으면 None"""
    for i, key in enumerate(calls):
        if i >= len(ch["path"]) - 1 or key != ch["path"][i]:
            return i + 1
    return None

def run_chains(client, n, chains):
    out = []
    for ch in chains:
        tr = run_agent(client, SYSTEM_CHAIN, chain_user(n), chain_tools(ch), max_steps=n + CFG["extra_steps"], temperature=CFG["temperature"], seed=CFG["seed"], max_tokens=CFG["max_tokens"])
        keys = [str(c["args"].get("key")) for st in tr.steps for c in st["calls"]]
        errs = sum(1 for st in tr.steps for c in st["calls"] if c["obs"].startswith("오류"))
        out.append({"n": n, "성공": bool(score_chain(ch, tr.final)), "스텝": tr.n_steps, "조회": len(keys), "오류 관찰": errs, "첫 이탈": first_dev(ch, keys),
                    "끝난 이유": tr.stopped, "토큰": tr.trace.tokens, "traj": tr})
    return out

ROWS, CH = [], None
try:
    for n in CFG["lengths"]:
        ROWS += run_chains(llm, n, CHAINS[n])
    CH = pd.DataFrame([{k: v for k, v in r.items() if k != "traj"} for r in ROWS])
    print(f"\\n모델 {llm.model} · 사슬 {len(CH)}개 실행 (서버 호출 {llm.n_live}번 · 캐시 {llm.n_cache}번)")
    print(CH.groupby("n").agg(성공=("성공", "sum"), 사슬=("성공", "size"), 평균_스텝=("스텝", "mean"), 평균_조회=("조회", "mean"), 평균_오류관찰=("오류 관찰", "mean")).to_string(float_format=lambda v: f"{v:.2f}"))
    pick = next((r for r in ROWS if r["n"] == 4), ROWS[0])
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.5 * (2 + 3 * pick["traj"].n_steps))))
    draw_trajectory(pick["traj"], title=f"궤적 하나 (길이 {pick['n']} · 성공 {pick['성공']})", ax=ax, wrap=30)
    plt.tight_layout(); plt.show()
except (LLMUnavailable, CacheMiss) as e:
    print("이 셀은 LM Studio 서버(또는 강사 캐시)가 있어야 돈다.\\n", e, "\\n→ 셀 5 · 6 · 7 은 안내만 내고 넘어간다. 셀 2 · 3 의 결과는 그대로 읽을 수 있다.")
''')

b.md('''
### 셀 5 · 실측 곡선 vs 이론
- 길이마다 성공률을 구하고, 스텝 하나의 성공률을 `p̂ = 성공률^(1/n)` 으로 거꾸로 추정한다. 스텝이 독립이고 복구가 없다면 p̂ 는 길이와 무관하게 같은 값이어야 한다
- 이론 곡선 두 가지를 그린다 : (a) 길이별 p̂ 의 평균으로 그린 `p̂^n`, (b) 길이 1 의 p̂ 하나만 쓴 `p̂₁^n`(짧은 과제만 보고 긴 과제를 예측한 것). 점의 막대는 95% 구간(윌슨)이다. 사슬이 길이마다 적으므로 구간이 넓다
- 오른쪽은 길이별 p̂. 위치별로 「직전까지 경로를 지킨 사슬 중 이번 호출에서 벗어난 비율」도 표로 찍는다
- 실제 곡선이 이론보다 빠르게 / 느리게 떨어졌는지는 숫자를 보고 확인 질문에서 따진다
''')
b.code('''
# 셀 5 · 실측 곡선 vs 이론
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

G = None
if CH is None:
    print("셀 4 가 끝나지 않아 이 셀은 넘어간다.")
else:
    G = CH.groupby("n").agg(성공수=("성공", "sum"), 사슬수=("성공", "size"), 평균스텝=("스텝", "mean"), 평균오류관찰=("오류 관찰", "mean")).reset_index()
    G["성공률"] = G["성공수"] / G["사슬수"]
    G["p̂"] = G["성공률"] ** (1 / G["n"])
    G["구간 하한"], G["구간 상한"] = zip(*[wilson(int(k), int(m)) for k, m in zip(G["성공수"], G["사슬수"])])
    p_mean, p_one = float(G["p̂"].mean()), float(G.loc[G["n"] == G["n"].min(), "p̂"].iloc[0])
    G["이론 p̂평균^n"] = p_mean ** G["n"]
    G["이론 p̂₁^n"] = p_one ** G["n"]
    print(f"설정 : 온도 {CFG['temperature']} · 시드 {CFG['seed']} · 길이별 사슬 {CFG['n_chains']}개 · 모델 {llm.model}")
    print(G[["n", "성공수", "사슬수", "성공률", "구간 하한", "구간 상한", "p̂", "이론 p̂평균^n", "이론 p̂₁^n", "평균스텝", "평균오류관찰"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\\np̂ 평균 = {p_mean:.3f} · 길이 {int(G['n'].min())} 의 p̂ = {p_one:.3f}")
    # 위치별 이탈 : 직전까지 경로를 지킨 사슬 중 이번 호출에서 처음 벗어난 비율
    pos = []
    for i in range(1, int(max(CFG["lengths"])) + 1):
        alive = CH[(CH["n"] >= i) & (CH["첫 이탈"].isna() | (CH["첫 이탈"] >= i))]
        dev = alive[alive["첫 이탈"] == i]
        pos.append({"몇 번째 호출": i, "경로를 지키던 사슬": len(alive), "여기서 벗어남": len(dev), "비율": len(dev) / len(alive) if len(alive) else float("nan")})
    POS = pd.DataFrame(pos)
    print("\\n위치별 이탈(조회 횟수를 지키지 못해 일찍 끝난 경우는 빠진 호출 번호에서 「벗어남」으로 센다) :")
    print(POS.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    Nn = np.arange(1, int(max(CFG["lengths"])) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.plot(Nn, p_mean ** Nn, color="#444", lw=1.6, label=f"이론 · 길이별 추정 p 의 평균 {p_mean:.2f} 로 계산")
    a1.plot(Nn, p_one ** Nn, color="#3F6694", lw=1.6, ls="--", label=f"이론 · 길이 1 의 추정 p {p_one:.2f} 만으로 계산")
    sp = SIM[(SIM["조건"] == "복구 없음") & (SIM["p"] == CFG["sim_p_focus"]) & (SIM["n"] <= int(max(CFG["lengths"])))]
    a1.scatter(sp["n"], sp["시뮬레이션"], marker="x", color="#999", zorder=2, label=f"가짜 정책 시뮬레이션 p = {CFG['sim_p_focus']:.2f}(셀 3)")
    a1.errorbar(G["n"], G["성공률"], yerr=[G["성공률"] - G["구간 하한"], G["구간 상한"] - G["성공률"]], fmt="o", color="#D9822B", capsize=3, ms=7, zorder=4, label=f"실측 · {llm.model}")
    a1.set_xlabel("사슬 길이 n (조회 횟수)"); a1.set_ylabel("끝까지 성공한 비율"); a1.set_ylim(0, 1.05); a1.legend(frameon=False, fontsize=8)
    a1.set_title("이론 · 시뮬레이션 · 실측", fontsize=11, loc="left")
    lo = [lo_ ** (1 / n) for lo_, n in zip(G["구간 하한"], G["n"])]
    hi = [hi_ ** (1 / n) for hi_, n in zip(G["구간 상한"], G["n"])]
    xs = np.arange(len(G))
    a2.bar(xs, G["p̂"], yerr=[G["p̂"] - lo, np.array(hi) - G["p̂"]], capsize=3, color=COLOR["llm"], ec="#444", lw=0.8)
    for xi, v in zip(xs, G["p̂"]):
        a2.text(xi, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(xs); a2.set_xticklabels([f"n={n}" for n in G["n"]], fontsize=9)
    a2.set_ylim(0, 1.08); a2.set_ylabel("단계별 성공률 추정 p")
    a2.set_title("길이별 추정 p (독립이면 길이와 무관)", fontsize=11, loc="left")
    plt.tight_layout(); plt.show()
''')

b.md('''
### 셀 6 · 모델 크기 비교 ① — 지금 로드된 모델을 측정해 캐시에 쌓는다
- 세 지표를 모델마다 같은 방식으로 얻는다
  - **유효한 도구 호출 비율** : 도구 호출이 JSON 으로 해석되고, 도구 이름이 목록에 있고, 필수 인자가 모두 채워진 비율(첫 행동 시험 질문 + 길이 8 사슬의 첫 호출)
  - **첫 행동 정확도** : 과제 D 의 「주문번호」 · 「이름+상품」 질문에서 첫 도구 호출이 정답 도구인 비율(주문번호 → `lookup_order`, 이름+상품 → `search_customer`)
  - **길이 8 연쇄 성공률** : 셀 4 와 같은 규칙으로 채점
- **모델은 한 번에 하나만 서버에 로드된다.** 다른 크기의 모델을 보려면 LM Studio 에서 모델을 바꿔 로드한 뒤 이 셀을 다시 돌린다 — 응답은 모델 이름별 `cache/` 폴더에 쌓이고, 이미 쌓인 모델만 다음 셀의 표에 들어간다
- 모델 이름은 환경변수 `AGENT_MODEL` 로 바꾼다(LM Studio 에 로드한 이름과 같아야 한다). 이 셀은 캐시에 있으면 서버를 부르지 않는다
''')
b.code('''
# 셀 6 · 지금 로드된 모델을 측정해 캐시에 쌓는다
def valid_call(rep, tools):
    """도구 호출이 모두 (JSON 으로 해석됨 · 목록에 있는 이름 · 필수 인자가 비어 있지 않음) 인가. 도구 호출이 아예 없으면 유효하지 않다"""
    if not rep.tool_calls:
        return False
    by = {t.name: t for t in tools}
    for c in rep.tool_calls:
        t = by.get(c["name"])
        if t is None or not c["valid"]:
            return False
        if not all(k in c["arguments"] and str(c["arguments"][k]).strip() for k in t.parameters.get("required", [])):
            return False
    return True

def measure_model(client):
    schemas_D = [t.schema() for t in TOOLS_D]
    valid, first = [], []
    for q in QD:
        r = client.chat([{"role": "system", "content": SYSTEM_D}, {"role": "user", "content": q["q"]}], tools=schemas_D, temperature=CFG["temperature"], seed=CFG["seed"], max_tokens=CFG["max_tokens"])
        valid.append(valid_call(r, TOOLS_D))
        want = "lookup_order" if q["유형"] == "주문번호" else "search_customer"
        first.append(bool(r.tool_calls) and r.tool_calls[0]["name"] == want)
    for ch in CHAINS[N_CMP]:
        r = client.chat([{"role": "system", "content": SYSTEM_CHAIN}, {"role": "user", "content": chain_user(N_CMP)}], tools=[t.schema() for t in chain_tools(ch)],
                        temperature=CFG["temperature"], seed=CFG["seed"], max_tokens=CFG["max_tokens"])
        valid.append(valid_call(r, chain_tools(ch)))
    chain = run_chains(client, N_CMP, CHAINS[N_CMP])
    return {"유효한 도구 호출 비율": float(np.mean(valid)), "첫 행동 정확도": float(np.mean(first)), f"길이 {N_CMP} 연쇄 성공률": float(np.mean([r["성공"] for r in chain])),
            "호출 수(유효 판정)": len(valid), "첫 행동 질문 수": len(first), "사슬 수": len(chain)}

MEAS = {}
try:
    MEAS[llm.model] = measure_model(llm)
    print(f"모델 {llm.model} 측정 완료 — 캐시 폴더 : {llm.dir}")
    print(json.dumps(MEAS[llm.model], ensure_ascii=False, indent=1))
except (LLMUnavailable, CacheMiss) as e:
    print("이 셀은 LM Studio 서버(또는 강사 캐시)가 있어야 돈다.\\n", e, "\\n→ 셀 7 은 이미 캐시가 쌓인 모델만 모은다.")
''')

b.md('''
### 셀 7 · 모델 크기 비교 ② — 캐시가 쌓인 모델만 모은다
- `LLM(model=m, mode="replay")` 로 모델마다 캐시만 읽어 같은 세 지표를 집계한다. 캐시에 없는 호출이 하나라도 있으면 그 모델은 「아직 측정하지 않음」으로 표에서 뺀다 — 서버를 부르지 않는다
- 사슬 수 · 질문 수가 작아서 비율 하나에 걸리는 표본이 적다(첫 행동 질문 수와 사슬 수를 표에 함께 적는다). 모델 사이 차이가 표본 오차 안인지도 같이 본다
- 지표 세 개가 같은 크기에서 함께 오르는지, 어느 지표가 먼저 오르는지는 숫자를 그대로 보고 확인 질문에서 따진다
''')
b.code('''
# 셀 7 · 캐시가 쌓인 모델만 모아 표 · 그림
RESM = {}
for m in MODELS:
    try:
        RESM[m] = measure_model(LLM(model=m, mode="replay"))
    except CacheMiss:
        print(f"{m} : 캐시에 없다 → 아직 측정하지 않음(표에서 뺀다). 이 모델을 로드하고 셀 6 을 돌리면 쌓인다")
    except LLMUnavailable as e:
        print(m, ":", e)
if not RESM:
    print("\\n캐시가 쌓인 모델이 없다.")
else:
    T = pd.DataFrame(RESM).T
    print(f"\\n설정 : 온도 {CFG['temperature']} · 시드 {CFG['seed']} · 사슬 길이 {N_CMP}")
    print(T.to_string(float_format=lambda v: f"{v:.3f}"))
    cols = ["유효한 도구 호출 비율", "첫 행동 정확도", f"길이 {N_CMP} 연쇄 성공률"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.26
    xs = np.arange(len(T))
    for i, (c, col) in enumerate(zip(cols, [COLOR["code"], COLOR["tool"], COLOR["llm"]])):
        ax.bar(xs + (i - 1) * w, T[c], w, color=col, ec="#444", lw=0.8, label=c)
        for xi, v in zip(xs, T[c]):
            ax.text(xi + (i - 1) * w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels(list(T.index), fontsize=8.5)
    ax.set_ylim(0, 1.28); ax.set_ylabel("비율")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    ax.set_title("모델 크기별 세 지표 (캐시가 쌓인 모델만)", fontsize=11, loc="left")
    plt.tight_layout(); plt.show()
    if len(T) >= 1:
        p8 = T[f"길이 {N_CMP} 연쇄 성공률"] ** (1 / N_CMP)
        print("\\n길이 8 연쇄 성공률에서 거꾸로 추정한 단계별 성공률 p̂ :", {k: round(float(v), 3) for k, v in p8.items()})
''')

b.md('''
### 셀 8 · 정리 · 결과 저장
''')
b.code('''
# 셀 8 · 정리 · 결과 저장
RES = {"설정": {"모델": llm.model, **CFG, "비교 모델": MODELS, "비교 사슬 길이": N_CMP},
       "이론": {"0.95^20": 0.95 ** 20},
       "시뮬레이션": SIM.to_dict(orient="records"),
       "실측 길이별": (G.drop(columns=["이론 p̂평균^n", "이론 p̂₁^n"]).to_dict(orient="records") if G is not None else "측정하지 않음(서버 없음)"),
       "위치별 이탈": (POS.to_dict(orient="records") if G is not None else "측정하지 않음(서버 없음)"),
       "모델 크기 비교": RESM or "측정하지 않음(캐시 없음)"}
save_measured("04", RES)
print("스텝 성공률이 곱해져 길이에 따라 내려가는 모습과, 모델에 따라 지표가 달라지는 모습을 숫자로 남겼다. 해석은 아래 확인 질문에서 숫자를 보고 한다.")
''')

b.md('''
### 확인 질문
1. 셀 3 에서 시뮬레이션이 `p^n` 과 맞은 조건은 무엇이었는가(오류가 서로 독립인가, 복구가 있는가)? 복구가 있는 조건에서 곡선이 달라진 이유를 설명하고, 오류가 관찰에 보이지 않는 조건에서는 복구해도 소용이 없는 이유를 **검증 신호**와 연결해 말해 보자.
2. 셀 5 에서 실측 성공률이 이론 곡선보다 빠르게 떨어졌는가, 느리게 떨어졌는가? 길이별 p̂ 가 길이와 무관하게 같았는지, 위치별 이탈 비율이 뒤로 갈수록 달라졌는지를 숫자로 보고 「스텝이 독립이다」라는 가정이 어디서 어긋날 수 있는지 설명해 보자(표본 수가 적다는 점도 함께 고려한다).
3. 셀 7 에서 모델이 커질 때 세 지표 중 어느 것이 먼저 오르고 어느 것이 늦게 오르는가? 길이 8 연쇄 성공률은 유효한 도구 호출 비율 · 첫 행동 정확도에서 예상한 곱으로 설명되는가? 「능력 임계를 넘어야 자율 루프가 성립한다」는 말에 비추어, 임계가 보이는 지표와 보이지 않는 지표를 구분해 말해 보자.
''')

b.save()
