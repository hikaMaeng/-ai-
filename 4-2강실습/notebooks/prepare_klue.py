"""3부(퓨전·분포 진단) 실습용 KLUE-MRC 데이터를 준비한다. 00번 노트북이 백그라운드로 실행한다.

  python prepare_klue.py            # 이미 끝난 단계는 건너뛴다 (4-1 을 마쳤다면 4번만 한다)
  python prepare_klue.py --force    # 처음부터 다시

4강실습(4-1)의 코드를 그대로 재사용한다.
  1) 텍스트 적재      4강실습/loader/load.py     (docs 5,309 · questions 5,841)
  2) BM25 역색인     4강실습/sql/20~22 *.sql   (bm25_search 함수까지)
  3) 문서 임베딩      지문 통째 1개 벡터         ← 4-1 과 같은 방식 (앞 128토큰만 담긴다)
  4) 청크 임베딩      문장 경계로 250자 청크     ← 4-2 에서 추가. "제목 + 청크"를 임베딩
"""
import argparse
import os
import re
import time

import psycopg

import load  # 4강실습/loader/load.py (jupyter 이미지의 PYTHONPATH)

DSN = os.environ["DATABASE_URL"]
SQL41 = "/work/4강실습/sql"
SENT = re.compile(r"(?<=[.!?])\s+")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def one(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def chunk(text: str, max_chars: int = 250) -> list[str]:
    """문장 경계에서 자르고, max_chars 를 넘지 않게 문장을 이어 붙인다."""
    out, cur = [], ""
    for s in SENT.split(text.strip()):
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def step_text(conn, force):
    if not force and one(conn, "SELECT count(*) FROM docs") > 0:
        log("1) 텍스트: 이미 적재됨 → 건너뜀")
        conn.execute("CREATE TABLE IF NOT EXISTS eval_q AS SELECT id, doc_id, question FROM questions "
                     "WHERE id % 11 = 0 ORDER BY id LIMIT 500")   # 4-1 의 14번에서 만들었을 수도 있다
        conn.commit()
        return
    log("1) 텍스트 적재 (KLUE-MRC validation)")
    docs, qs = load.build_frames(load.download("validation"))
    load.load_text(conn, docs, qs)
    with conn.cursor() as cur:   # 4-1 의 14번과 같은 평가 샘플 500개
        cur.execute("DROP TABLE IF EXISTS eval_q; "
                    "CREATE TABLE eval_q AS SELECT id, doc_id, question FROM questions "
                    "WHERE id % 11 = 0 ORDER BY id LIMIT 500")
    conn.commit()


def step_bm25(conn, force):
    done = one(conn, "SELECT to_regproc('bm25_search') IS NOT NULL")
    if done and not force:
        log("2) BM25: 이미 구축됨 → 건너뜀")
        return
    log("2) BM25 역색인 구축 (4강실습/sql 20·21·22)")
    for f in ("20_bm25_tables.sql", "21_bm25_build.sql", "22_bm25_search.sql"):
        with open(f"{SQL41}/{f}", encoding="utf-8") as fp:
            conn.execute(fp.read())
        conn.commit()
        log(f"   {f} 완료")


def step_doc_embed(conn, force):
    if not force and one(conn, "SELECT count(*) FROM docs WHERE embedding IS NULL") == 0:
        log("3) 문서 임베딩: 이미 있음 → 건너뜀")
        return
    log("3) 문서 임베딩 (지문 통째, 4-1 과 동일)")
    load.embed_table(conn, "docs", "coalesce(title, '') || E'\\n' || content")
    load.embed_table(conn, "questions", "question")
    conn.execute("CREATE INDEX IF NOT EXISTS docs_embedding_hnsw ON docs USING hnsw (embedding vector_cosine_ops)")
    conn.commit()


def step_chunks(conn, force):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_chunks (
            id        bigserial PRIMARY KEY,
            doc_id    bigint NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
            chunk_no  int    NOT NULL,
            content   text   NOT NULL,
            embedding vector(384)
        )""")
    conn.commit()
    if not force and one(conn, "SELECT count(*) FROM doc_chunks") > 0 and \
            one(conn, "SELECT count(*) FROM doc_chunks WHERE embedding IS NULL") == 0:
        log("4) 청크 임베딩: 이미 있음 → 건너뜀")
        return
    log("4) 청크 분할 (문장 경계, 250자)")
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS doc_chunks_embedding_hnsw")
        cur.execute("TRUNCATE doc_chunks")
        cur.execute("SELECT id, content FROM docs ORDER BY id")
        rows = cur.fetchall()
        with cur.copy("COPY doc_chunks (doc_id, chunk_no, content) FROM STDIN") as cp:
            for doc_id, content in rows:
                for i, c in enumerate(chunk(content)):
                    cp.write_row((doc_id, i, c))
    conn.commit()
    n = one(conn, "SELECT count(*) FROM doc_chunks")
    log(f"   청크 {n:,}개 (지문당 평균 {n / len(rows):.1f}개)")
    # 청크만 떼어 놓으면 "이 회사는…"처럼 주어가 사라진다. 제목을 앞에 붙여 맥락을 보탠다
    # (슬라이드의 Contextual Retrieval 을 가장 단순하게 흉내 낸 것)
    load.embed_table(conn, "doc_chunks",
                     "(SELECT coalesce(d.title, '') FROM docs d WHERE d.id = doc_chunks.doc_id)"
                     " || E'\\n' || content")
    conn.execute("CREATE INDEX IF NOT EXISTS doc_chunks_embedding_hnsw "
                 "ON doc_chunks USING hnsw (embedding vector_cosine_ops)")
    conn.execute("ANALYZE doc_chunks")
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t = time.time()
    with psycopg.connect(DSN) as conn:
        step_text(conn, args.force)
        step_bm25(conn, args.force)
        step_doc_embed(conn, args.force)
        step_chunks(conn, args.force)
    log(f"✔ KLUE 준비 완료 ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
