"""4-2강 실습 공용 도구. 노트북 첫 셀에서 `from labutil import *` 로 불러 쓴다."""
import os
import re
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector

DSN = os.environ.get("DATABASE_URL", "postgresql://lab:lab@db:5432/lab")

# ── 그래프 ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "NanumGothic",       # 한글 글꼴 (jupyter 이미지에 설치됨)
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
pd.set_option("display.max_colwidth", 80)


# ── DB ──────────────────────────────────────────────────────────────
def connect():
    """autocommit 연결. numpy 배열을 vector 타입으로 바로 주고받을 수 있게 등록한다."""
    conn = psycopg.connect(DSN, autocommit=True)
    # psycopg 는 같은 SQL 을 5번 넘게 실행하면 서버에 prepare 해 두고 실행 계획을 재사용한다.
    # 그러면 SET hnsw.ef_search / enable_indexscan 을 바꿔도 옛 계획이 쓰이므로 끈다.
    conn.prepare_threshold = None
    register_vector(conn)
    return conn


def q(conn, sql, params=None):
    """SELECT 결과를 DataFrame 으로."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description is None:
            return None
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def q1(conn, sql, params=None):
    """값 하나."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def explain(conn, sql, params=None):
    """EXPLAIN (ANALYZE) 결과를 문자열로."""
    with conn.cursor() as cur:
        cur.execute("EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF) " + sql, params)
        text = "\n".join(r[0] for r in cur.fetchall())
    return re.sub(r"'\[[^\]]{40,}\]'", "'[…]'", text)      # 긴 벡터 리터럴은 줄여서 보여 준다


def rel_size(conn, name):
    """테이블·인덱스 크기(MB)."""
    return q1(conn, "SELECT pg_relation_size(%s::regclass)", (name,)) / 2**20


@contextmanager
def timer(label=""):
    t = time.perf_counter()
    box = {}
    yield box
    box["s"] = time.perf_counter() - t
    if label:
        print(f"{label}: {box['s']:.2f}s")


# ── 최근접 이웃 ───────────────────────────────────────────────────────
def brute_knn(X, Q, k):
    """정확한 k-NN (L2). 모든 ANN 실험의 정답지. 반환: (len(Q), k) 인덱스."""
    X = np.asarray(X, dtype=np.float32)
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float32))
    # ‖q−x‖² = ‖q‖² − 2q·x + ‖x‖²  (q 마다 상수인 ‖q‖² 는 순위에 영향 없음)
    d = (X ** 2).sum(1)[None, :] - 2 * Q @ X.T
    idx = np.argpartition(d, k, axis=1)[:, :k]
    order = np.take_along_axis(d, idx, 1).argsort(1)
    return np.take_along_axis(idx, order, 1)


def recall(found, truth):
    """ANN recall@k = (찾은 것 ∩ 정답) / k 의 평균. found·truth 는 질의별 id 목록."""
    return float(np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)]))


def pg_bench(conn, table, ds, k=10, where="", n=None, op="<->"):
    """20번에서 만든 질의(vec_query)를 table 에 던져 recall@k 와 지연시간을 잰다.
    세션 설정(SET hnsw.ef_search 등)은 호출 전에 한다. 반환: dict(recall, ms, 결과수)"""
    sql = f"SELECT id FROM {table} {where} ORDER BY emb {op} %(q)s LIMIT {k}"
    return pg_bench_sql(conn, sql, ds, k=k, n=n)


def pg_bench_sql(conn, sql, ds, k=10, n=None):
    """임의의 SQL 로 recall@k · 지연시간 측정. SQL 안의 %(q)s 자리에 질의 벡터가 들어간다."""
    qs = q(conn, "SELECT qid, emb FROM vec_query WHERE ds = %s ORDER BY qid", (ds,))
    if n:
        qs = qs.head(n)
    gt = q(conn, "SELECT qid, array_agg(id ORDER BY rank) ids FROM vec_gt WHERE ds = %s AND rank <= %s GROUP BY qid",
           (ds, k)).set_index("qid")["ids"]
    found, times = [], []
    with conn.cursor() as cur:
        for emb in qs["emb"][:20]:                 # 예열: 인덱스 페이지를 캐시에 올린다 (측정 제외)
            cur.execute(sql, {"q": emb}); cur.fetchall()
        for qid, emb in qs.itertuples(index=False):
            t = time.perf_counter()
            cur.execute(sql, {"q": emb})
            ids = [r[0] for r in cur.fetchall()]
            times.append(time.perf_counter() - t)
            found.append(ids)
    return {"recall": recall(found, [gt[qid] for qid in qs["qid"]]),
            "ms": 1000 * float(np.median(times)),
            "결과수": float(np.mean([len(f) for f in found]))}


def make_clusters(n, d, n_clusters, spread=1.0, center_scale=4.0, seed=0):
    """군집형 합성 데이터: 중심을 뽑고 가우시안 잡음을 더한다. 반환: (X, 라벨)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, center_scale, (n_clusters, d)).astype(np.float32)
    labels = rng.integers(0, n_clusters, n)
    X = centers[labels] + rng.normal(0, spread, (n, d)).astype(np.float32)
    return X.astype(np.float32), labels
