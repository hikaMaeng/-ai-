import pickle, numpy as np
runs = pickle.load(open("/work/4-2/notebooks/results/klue_runs.pkl", "rb"))
Q = sorted(runs["gold"]); rng = np.random.default_rng(0)

def rr_vec(ranked):          # 질문별 역순위(상위10 밖이면 0)
    out = []
    for q in Q:
        lst = list(ranked[q])[:10]; g = runs["gold"][q]
        out.append(1 / (lst.index(g) + 1) if g in lst else 0.0)
    return np.array(out)

def rrf(k=60, w=(1, 1)):
    out = {}
    for q in Q:
        s = {}
        for L, wi in zip(["bm25", "vec"], w):
            for r, (d, _) in enumerate(runs[L][q], 1): s[d] = s.get(d, 0) + wi / (k + r)
        out[q] = sorted(s, key=s.get, reverse=True)
    return out

def mm(alpha):
    out = {}
    for q in Q:
        f = {}
        for L, wg in [("bm25", alpha), ("vec", 1 - alpha)]:
            sc = np.array([s for _, s in runs[L][q]], float)
            if len(sc) == 0: continue
            lo, hi = sc.min(), sc.max()
            for d, s in runs[L][q]: f[d] = f.get(d, 0) + wg * ((s - lo) / (hi - lo) if hi > lo else 1)
        out[q] = sorted(f, key=f.get, reverse=True)
    return out

bm = rr_vec({q: [d for d, _ in runs["bm25"][q]] for q in Q})
RRF = {k: rr_vec(rrf(k)) for k in [1, 5, 10, 20, 40, 60, 100, 200]}
WR = {w: rr_vec(rrf(60, (w, 1))) for w in [0.5, 1, 1.5, 2, 3, 5]}
MM = {a: rr_vec(mm(a)) for a in np.round(np.linspace(0, 1, 11), 1)}
f = lambda r: f"Hit {np.mean(r > 0):.3f} · MRR {r.mean():.3f}"

print("① 파라미터를 고르지 않은 기본값끼리")
print("  BM25            ", f(bm))
print("  RRF k=60        ", f(RRF[60]))
print("  Min-Max α=0.5   ", f(MM[0.5]))
print("  Min-Max 전체 α:", {a: round(r.mean(), 3) for a, r in MM.items()})
print("  RRF 전체 k:", {k: round(r.mean(), 3) for k, r in RRF.items()})

print("\n② 반씩 나눠 한쪽에서 파라미터를 고르고 다른 쪽에서 채점 (무작위 분할 500회 평균)")
res = {"RRF k 선택": [], "가중 RRF w 선택": [], "Min-Max α 선택": [], "BM25": [], "RRF k=60 고정": []}
hit = {k: [] for k in res}
n = len(Q)
for _ in range(500):
    idx = rng.permutation(n); tr, te = idx[: n // 2], idx[n // 2:]
    for name, fam in [("RRF k 선택", RRF), ("가중 RRF w 선택", WR), ("Min-Max α 선택", MM)]:
        best = max(fam, key=lambda p: fam[p][tr].mean())
        res[name].append(fam[best][te].mean()); hit[name].append(np.mean(fam[best][te] > 0))
    res["BM25"].append(bm[te].mean()); hit["BM25"].append(np.mean(bm[te] > 0))
    res["RRF k=60 고정"].append(RRF[60][te].mean()); hit["RRF k=60 고정"].append(np.mean(RRF[60][te] > 0))
for k in res: print(f"  {k:14s} 시험 절반 Hit {np.mean(hit[k]):.3f} · MRR {np.mean(res[k]):.3f}")

print("\n③ 표의 '최고' 끼리 차이가 질문 몇 개에서 나오나 (Min-Max α=0.6 vs RRF k=1)")
a, b = MM[0.6], RRF[1]
print("  Hit 차이 질문 수 : Min-Max만 맞힘", int(((a > 0) & (b == 0)).sum()), "· RRF만 맞힘", int(((b > 0) & (a == 0)).sum()))
print("  순위가 달라진 질문", int((a != b).sum()), "개 · Min-Max가 나은", int((a > b).sum()), "· RRF가 나은", int((b > a).sum()))
d = a - b; bs = [d[rng.integers(0, n, n)].mean() for _ in range(5000)]
print(f"  MRR 차이 {d.mean():+.3f}, 부트스트랩 95% 구간 [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
for nm, other in [("Min-Max α=0.6 − BM25", bm), ("Min-Max α=0.6 − RRF k=60", RRF[60])]:
    d = a - other; bs = [d[rng.integers(0, n, n)].mean() for _ in range(5000)]
    print(f"  {nm}: MRR {d.mean():+.3f}, 95% [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] · Hit 차이 {int(((a > 0) & (other == 0)).sum())} vs {int(((other > 0) & (a == 0)).sum())}")
