import pickle, numpy as np, pandas as pd, psycopg

conn = psycopg.connect("postgresql://lab:lab@db:5432/lab")
pd.set_option("display.width", 220)
runs = pickle.load(open("/work/4-2/notebooks/results/klue_runs.pkl", "rb"))
Q = sorted(runs["gold"])


def rrf(q, k):
    s = {}
    for L in ["bm25", "vec"]:
        for r, (d, _) in enumerate(runs[L][q], 1): s[d] = s.get(d, 0) + 1 / (k + r)
    return sorted(s, key=s.get, reverse=True)


def mm(q, a):
    f = {}
    for L, w in [("bm25", a), ("vec", 1 - a)]:
        sc = np.array([s for _, s in runs[L][q]], float)
        if len(sc) == 0: continue
        lo, hi = sc.min(), sc.max()
        for d, s in runs[L][q]: f[d] = f.get(d, 0) + w * ((s - lo) / (hi - lo) if hi > lo else 1)
    return sorted(f, key=f.get, reverse=True)


def rr(lst, g):
    lst = list(lst)[:10]
    return 1 / (lst.index(g) + 1) if g in lst else 0.0


meta = pd.DataFrame(conn.execute("""
  SELECT q.id, q.question_type, length(d.content),
         CASE WHEN q.answer IS NOT NULL AND strpos(d.content, q.answer) > 0
              THEN strpos(d.content, q.answer)::float / length(d.content) END,
         d.source,
         (SELECT count(*) FROM unnest(tsvector_to_array(to_tsvector('korean', q.question))) w
            WHERE w = ANY (tsvector_to_array(d.tsv)))::float
           / greatest(array_length(tsvector_to_array(to_tsvector('korean', q.question)), 1), 1)
  FROM questions q JOIN docs d ON d.id = q.doc_id WHERE q.id = ANY(%s)""", (Q,)).fetchall(),
  columns=["qid", "유형", "chars", "ans_pos", "출처", "overlap"]).set_index("qid")

M = {"BM25": lambda q: [d for d, _ in runs["bm25"][q]], "벡터": lambda q: [d for d, _ in runs["vec"][q]],
     "RRF k=60": lambda q: rrf(q, 60), "RRF k=1": lambda q: rrf(q, 1),
     "MM α=0.4": lambda q: mm(q, 0.4), "MM α=0.6": lambda q: mm(q, 0.6), "MM α=0.8": lambda q: mm(q, 0.8)}
for name, f in M.items():
    meta[name] = [rr(f(q), runs["gold"][q]) for q in meta.index]

meta["겹침"] = pd.cut(meta["overlap"], [-0.01, 0.5, 0.75, 0.99, 1.0], labels=["겹침~50%", "겹침50~75", "겹침75~99", "겹침100%"])
meta["길이"] = pd.cut(meta["chars"], [0, 600, 900, 1200, 3000], labels=["~600자", "600~900", "900~1200", "1200~"])
meta["위치"] = pd.cut(meta["ans_pos"], [0, 0.25, 0.5, 0.75, 1.0], labels=["답 앞1/4", "답~1/2", "답~3/4", "답 끝1/4"])
meta["유형"] = meta["유형"].map({1: "유형1 패러프레이즈", 2: "유형2 다문장", 3: "유형3 답없음"})
names = list(M)

for metric, fn in [("Hit@10", lambda s: (s > 0).mean()), ("MRR", lambda s: s.mean())]:
    rows = []
    for col in ["유형", "겹침", "길이", "위치", "출처"]:
        for key, g in meta.groupby(col, observed=True):
            r = {"구간": key, "n": len(g), **{n: round(fn(g[n]), 3) for n in names}}
            best = max(names, key=lambda n: r[n]); top = [n for n in names if r[n] == r[best]]
            r["1위"] = " / ".join(top)
            rows.append(r)
    T = pd.DataFrame(rows).set_index("구간")
    print(f"\n=== {metric} ===")
    print(T.to_string())

# MM α=0.6 이 구간 1위가 아닌 곳에서 차이가 몇 문항인가
print("\n=== MM α=0.6 이 Hit@10 에서 진 구간 : 문항 수 차이 ===")
for col in ["유형", "겹침", "길이", "위치", "출처"]:
    for key, g in meta.groupby(col, observed=True):
        mmh = int((g["MM α=0.6"] > 0).sum())
        for n in names:
            h = int((g[n] > 0).sum())
            if h > mmh: print(f"  {key} (n={len(g)}): {n} {h} vs MM0.6 {mmh}")
