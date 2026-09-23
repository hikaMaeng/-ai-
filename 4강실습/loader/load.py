"""허깅페이스 KLUE-MRC 데이터셋을 받아 pgvector 에 적재한다.

  docker compose run --rm loader                              # validation 분할 (지문 5,309 / 질문 5,841)
  docker compose run --rm loader --no-embed                   # 텍스트만 (1분 이내)
  docker compose run --rm loader --embed-only                 # 이미 적재된 텍스트에 임베딩만
  docker compose run --rm loader --splits train validation    # 전체 (지문 약 1.9만, 임베딩 15분+)

흐름
  1) parquet 다운로드 → 지문 중복 제거
  2) COPY 로 docs / questions 적재 → 커밋        ← 여기서부터 ts_rank·BM25 실습 가능
  3) 임베딩 계산 → UPDATE 로 채움 → HNSW 인덱스  ← 벡터 실습(25번)에만 필요, 뒤에서 계속 진행
"""
import argparse
import hashlib
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import psycopg
import requests

DATASET = "klue/klue"
CONFIG = "mrc"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384차원, 입력 128토큰
warnings.filterwarnings("ignore", category=UserWarning)   # fastembed 풀링 방식 안내 경고
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

HF_HOME = Path(os.environ.get("HF_HOME", "/cache/hf"))


def download(split: str) -> pd.DataFrame:
    """허깅페이스가 자동 변환해 둔 parquet 파일을 받는다 (datasets 라이브러리 없이)."""
    path = HF_HOME / "parquet" / f"{CONFIG}-{split}.parquet"
    if not path.exists():
        url = f"https://huggingface.co/api/datasets/{DATASET}/parquet/{CONFIG}/{split}/0.parquet"
        print(f"  다운로드: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_parquet(path)
    print(f"  {split}: {len(df):,} 질문")
    return df


def build_frames(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """질문 단위 원본을 지문(docs)과 질문(questions)으로 정규화한다.
    같은 지문에 질문이 여러 개 달려 있으므로 지문 내용의 md5 로 중복을 제거한다."""
    raw = raw.copy()
    raw["doc_key"] = raw["context"].map(lambda s: hashlib.md5(s.encode()).hexdigest())

    docs = (raw.drop_duplicates("doc_key")
               .loc[:, ["doc_key", "title", "context", "news_category", "source"]]
               .rename(columns={"context": "content", "news_category": "category"})
               .reset_index(drop=True))
    docs["category"] = docs["category"].replace({"": None})

    qs = raw.loc[:, ["guid", "doc_key", "question", "question_type", "is_impossible"]].copy()
    qs["answer"] = raw["answers"].map(lambda a: a["text"][0] if len(a["text"]) else None)
    return docs, qs


def load_text(conn, docs: pd.DataFrame, qs: pd.DataFrame):
    with conn.cursor() as cur:
        # BM25 트리거(7단계)가 있으면 DELETE 가 역색인도 함께 비운다.
        # TRUNCATE 는 행 삭제 트리거를 발동시키지 않으므로 DELETE 를 쓴다.
        cur.execute("DROP INDEX IF EXISTS docs_embedding_hnsw")
        cur.execute("DELETE FROM docs")
        t = time.time()
        with cur.copy("COPY docs (doc_key, title, content, category, source) FROM STDIN") as cp:
            for d in docs.itertuples(index=False):
                cp.write_row((d.doc_key, d.title, d.content, d.category, d.source))
        cur.execute("SELECT doc_key, id FROM docs")
        ids = dict(cur.fetchall())
        with cur.copy("COPY questions (guid, doc_id, question, answer, question_type, is_impossible) FROM STDIN") as cp:
            for q in qs.itertuples(index=False):
                cp.write_row((q.guid, ids[q.doc_key], q.question, q.answer,
                              int(q.question_type), bool(q.is_impossible)))
        cur.execute("ANALYZE docs; ANALYZE questions")
        cur.execute("SELECT count(*), sum(length(tsv)) FROM docs")
        n, lex = cur.fetchone()
    conn.commit()
    print(f"  docs {n:,}행 / questions {len(qs):,}행, {time.time() - t:.1f}s "
          f"(tsvector 는 생성 컬럼이라 COPY 하는 동안 형태소 분석까지 끝난다)")


def embed_table(conn, table: str, text_sql: str):
    """table 의 텍스트를 임베딩해 embedding 컬럼을 채운다."""
    from fastembed import TextEmbedding
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, {text_sql} FROM {table} ORDER BY id")
        rows = cur.fetchall()
    model = TextEmbedding(EMBED_MODEL, cache_dir=str(HF_HOME / "fastembed"))
    t = time.time()
    vecs = model.embed([r[1] for r in rows], batch_size=64)
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE _emb (id bigint, embedding vector(384)) ON COMMIT DROP")
        with cur.copy("COPY _emb (id, embedding) FROM STDIN") as cp:
            for i, ((id_, _), v) in enumerate(zip(rows, vecs), 1):
                cp.write_row((id_, "[" + ",".join(f"{x:.6f}" for x in v) + "]"))
                if i % 1000 == 0:
                    print(f"    {table} {i:,}/{len(rows):,} ({time.time() - t:.0f}s)")
        cur.execute(f"UPDATE {table} x SET embedding = e.embedding FROM _emb e WHERE x.id = e.id")
    conn.commit()
    print(f"  {table} 임베딩 {len(rows):,}건 {time.time() - t:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation"], choices=["train", "validation"])
    ap.add_argument("--no-embed", action="store_true", help="텍스트만 적재")
    ap.add_argument("--embed-only", action="store_true", help="적재된 텍스트에 임베딩만 추가")
    ap.add_argument("--limit", type=int, default=0, help="지문 수 제한(테스트용)")
    args = ap.parse_args()
    started = time.time()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        if not args.embed_only:
            print("1) 데이터셋 다운로드")
            raw = pd.concat([download(s) for s in args.splits], ignore_index=True)
            docs, qs = build_frames(raw)
            if args.limit:
                docs = docs.head(args.limit)
                qs = qs[qs["doc_key"].isin(docs["doc_key"])]
            print(f"  → 지문 {len(docs):,}개 / 질문 {len(qs):,}개")

            print("2) 텍스트 적재 (COPY)")
            load_text(conn, docs, qs)
            print(f"   ✔ 텍스트 적재 완료 ({time.time() - started:.0f}s). "
                  "지금부터 pgAdmin 에서 10~24번 실습을 시작해도 된다.")

        if not args.no_embed:
            print(f"3) 임베딩 ({EMBED_MODEL}) — 25번 실습에만 필요, 끝날 때까지 창을 닫지 말 것")
            embed_table(conn, "docs", "coalesce(title, '') || E'\\n' || content")
            embed_table(conn, "questions", "question")
            print("4) HNSW 인덱스 생성 (코사인 거리)")
            with conn.cursor() as cur:
                cur.execute("CREATE INDEX IF NOT EXISTS docs_embedding_hnsw "
                            "ON docs USING hnsw (embedding vector_cosine_ops)")
            conn.commit()

    print(f"완료 ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
