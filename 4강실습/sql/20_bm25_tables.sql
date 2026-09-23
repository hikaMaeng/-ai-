-- ==================================================================
-- 실습 7-1. BM25 를 위한 역색인 테이블 설계
--
--   BM25(D, Q) = Σ_{t∈Q} IDF(t) · f(t,D)·(k1+1) / ( f(t,D) + k1·(1 − b + b·|D|/avgdl) )
--   IDF(t)     = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )        -- Lucene 변형
--
--   공식에 필요한 값          →  저장할 테이블
--   f(t,D)  문서별 단어 빈도  →  bm25_tf     (term, doc_id, tf)   ← 역색인(inverted index)
--   |D|     문서 길이         →  bm25_doclen (doc_id, len)
--   N, avgdl                 →  bm25_stats  (n_docs, total_len)  ← 1행짜리 전역 통계
--   df(t)   단어가 나온 문서 수 →  bm25_df     (term, df)
--   IDF(t)                   →  bm25_idf    (term, idf)          ← df 와 N 으로 계산해 둔 캐시
--
--   모든 값은 docs.tsv(tsvector) 에서 뽑는다. 형태소 분석은 이미 끝나 있으므로
--   BM25 는 "어떻게 세고, 어떻게 점수를 매기는가"만 구현하면 된다.
-- ==================================================================

DROP TABLE IF EXISTS bm25_tf, bm25_doclen, bm25_df, bm25_idf, bm25_stats CASCADE;

-- 역색인: 질의어(term)로 찾아 들어가므로 PK 순서가 (term, doc_id)
CREATE TABLE bm25_tf (
    term   text   NOT NULL,
    doc_id bigint NOT NULL,
    tf     int    NOT NULL,
    PRIMARY KEY (term, doc_id)
);
CREATE INDEX bm25_tf_doc ON bm25_tf(doc_id);   -- 문서 삭제·수정 시 해당 문서의 행을 찾기 위함

CREATE TABLE bm25_doclen (
    doc_id bigint PRIMARY KEY,
    len    int    NOT NULL
);

CREATE TABLE bm25_df (
    term text PRIMARY KEY,
    df   int  NOT NULL
);

CREATE TABLE bm25_idf (
    term text   PRIMARY KEY,
    idf  float8 NOT NULL
);

CREATE TABLE bm25_stats (
    id        int    PRIMARY KEY DEFAULT 1 CHECK (id = 1),   -- 항상 1행
    n_docs    bigint NOT NULL DEFAULT 0,
    total_len bigint NOT NULL DEFAULT 0
);
INSERT INTO bm25_stats DEFAULT VALUES;

-- tsvector 를 풀어 보면 BM25 재료가 다 들어 있다
--   lexeme = term, cardinality(positions) = tf
SELECT d.id AS doc_id, t.lexeme AS term, cardinality(t.positions) AS tf, t.weights
FROM docs d, unnest(d.tsv) t
WHERE d.id = 1
ORDER BY tf DESC
LIMIT 15;

-- 주의: tsvector 는 형태소 하나당 위치를 최대 256개까지만 저장한다.
--       이 코퍼스(지문 ~1천자)에서는 문제없지만, 아주 긴 문서라면 tf 가 256 에서 잘린다.
SELECT max(cardinality(t.positions)) AS 최대_tf FROM docs d, unnest(d.tsv) t;
