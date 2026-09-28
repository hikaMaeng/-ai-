import pickle, requests, numpy as np, pandas as pd, psycopg
from pgvector.psycopg import register_vector

# LM Studio 주소 · 모델 이름 : 자기 환경 값으로 (아래는 강사 PC. 4-1 실습1 셀 1 과 같게)
URL, M = "http://host.docker.internal:12345/v1", "qwen3-embedding-8b:mp"
conn = psycopg.connect("postgresql://lab:lab@db:5432/lab"); register_vector(conn)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
cos = lambda a, b: float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def emb(x):
    r = requests.post(f"{URL}/embeddings", json={"model": M, "input": [x]}, timeout=120); r.raise_for_status()
    return np.array(r.json()["data"][0]["embedding"])


# ① 잘림 검사 : 가장 긴 지문
did, title, content, stored = conn.execute(
    "SELECT id, title, content, embedding FROM docs ORDER BY length(content) DESC LIMIT 1").fetchone()
full = (title or "") + "\n" + content
e_full = emb(full)
print("① 잘림 검사 — 지문", did, len(full), "자")
print("   저장된 임베딩 vs 지금 계산          ", round(cos(stored, e_full), 4))
print("   전문 vs 끝에 한 문장 덧붙임          ", round(cos(e_full, emb(full + "\n이 문장은 끝에 덧붙인 시험용 문장이다. 고양이 사료 추천.")), 4))
print("   전문 vs 앞 절반만                    ", round(cos(e_full, emb(full[: len(full) // 2])), 4))

# ② 질문별 BM25 · 벡터 순위와 속성
runs = pickle.load(open("/work/4-2/notebooks/results/klue_runs.pkl", "rb"))
Q = sorted(runs["gold"])
meta = pd.DataFrame(conn.execute("""
  SELECT q.id, q.question_type, q.answer, d.id AS gold, length(d.content) AS doclen_chars,
         CASE WHEN q.answer IS NOT NULL AND strpos(d.content, q.answer) > 0
              THEN strpos(d.content, q.answer)::float / length(d.content) END AS ans_pos,
         d.source,
         (SELECT count(*) FROM unnest(tsvector_to_array(to_tsvector('korean', q.question))) w
            WHERE w = ANY (tsvector_to_array(d.tsv)))::float
           / greatest(array_length(tsvector_to_array(to_tsvector('korean', q.question)), 1), 1) AS overlap
  FROM questions q JOIN docs d ON d.id = q.doc_id WHERE q.id = ANY(%s)""", (Q,)).fetchall(),
  columns=["qid", "qtype", "answer", "gold", "chars", "ans_pos", "source", "overlap"]).set_index("qid")


def rank(lst, g):
    lst = [d for d, _ in lst][:10]
    return lst.index(g) + 1 if g in lst else None


meta["bm25_hit"] = [rank(runs["bm25"][q], runs["gold"][q]) is not None for q in meta.index]
meta["vec_hit"] = [rank(runs["vec"][q], runs["gold"][q]) is not None for q in meta.index]
agg = lambda g: g.agg(질문수=("vec_hit", "size"), BM25_Hit=("bm25_hit", "mean"), 벡터_Hit=("vec_hit", "mean")).round(3)

print("\n② 질문 유형별 (1 패러프레이즈 · 2 다문장 추론 · 3 답 없음)")
print(agg(meta.groupby("qtype")))
print("\n   질문 형태소가 정답 지문에 들어 있는 비율(단어 겹침)")
meta["겹침"] = pd.cut(meta["overlap"], [-0.01, 0.5, 0.75, 0.99, 1.0], labels=["~50%", "50~75%", "75~99%", "100%"])
print(agg(meta.groupby("겹침", observed=True)))
print("\n   정답이 지문의 어디쯤에 있나 (답 문자열 위치 / 지문 길이)")
m2 = meta.dropna(subset=["ans_pos"]).copy()
m2["위치"] = pd.cut(m2["ans_pos"], [0, 0.25, 0.5, 0.75, 1.0], labels=["앞 1/4", "~1/2", "~3/4", "끝 1/4"])
print(agg(m2.groupby("위치", observed=True)))
print("\n   지문 길이")
meta["길이"] = pd.cut(meta["chars"], [0, 600, 900, 1200, 3000], labels=["~600자", "600~900", "900~1200", "1200~"])
print(agg(meta.groupby("길이", observed=True)))
print("\n   출처")
print(agg(meta.groupby("source")))

# ③ 벡터만 놓친 질문 : 벡터 1위는 무엇이었나
miss = [q for q in meta.index if meta.loc[q, "bm25_hit"] and not meta.loc[q, "vec_hit"]]
print(f"\n③ BM25 는 맞히고 벡터는 놓친 질문 {len(miss)}개 중 앞 6개")
titles = dict(conn.execute("SELECT id, left(title, 38) FROM docs").fetchall())
cat = dict(conn.execute("SELECT id, coalesce(category, source) FROM docs").fetchall())
same_cat = 0
for q in miss:
    g = runs["gold"][q]; top = runs["vec"][q][0][0]
    same_cat += cat[top] == cat[g]
for q in miss[:6]:
    g = runs["gold"][q]; top, s1 = runs["vec"][q][0]
    gs = dict(runs["vec"][q]).get(g)
    print(f"  Q: {runs['question'][q]}")
    print(f"     정답 [{g}] {titles[g]}  | 벡터 유사도 {gs if gs is None else round(gs, 3)}")
    print(f"     벡터1위 [{top}] {titles[top]}  | {s1:.3f}")
print(f"  벡터 1위가 정답과 같은 분야(category · 출처)인 비율 : {same_cat}/{len(miss)}")
