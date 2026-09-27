-- BM25 역색인 : 테이블 5개 · 함수 · 트리거 4개 (컨테이너 최초 기동 시 1회)
--
--   BM25(D, Q) = Σ_{t∈Q} IDF(t) · f(t,D)·(k1+1) / ( f(t,D) + k1·(1 − b + b·|D|/avgdl) )
--   IDF(t)     = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )        -- Lucene 변형
--
--   공식에 필요한 값            →  저장할 테이블
--   f(t,D)  문서별 단어 빈도    →  bm25_tf     (term, doc_id, tf)   ← 역색인(inverted index)
--   |D|     문서 길이           →  bm25_doclen (doc_id, len)
--   N, avgdl                   →  bm25_stats  (n_docs, total_len)  ← 1행짜리 전역 통계
--   df(t)   단어가 나온 문서 수  →  bm25_df     (term, df)
--   IDF(t)                     →  bm25_idf    (term, idf)          ← df 와 N 으로 계산해 둔 캐시
--
--   모든 값은 docs.tsv(형태소 분석 결과) 에서 뽑는다.
--   docs 에 INSERT · UPDATE · DELETE · TRUNCATE 가 일어나면 트리거가 이 테이블들을 자동으로 맞춘다.

-- ── 테이블 ──────────────────────────────────────────────────────────
CREATE TABLE bm25_tf (
    term   text   NOT NULL,
    doc_id bigint NOT NULL,
    tf     int    NOT NULL,
    PRIMARY KEY (term, doc_id)                   -- 질의어(term)로 찾아 들어가므로 term 이 앞
);
CREATE INDEX bm25_tf_doc ON bm25_tf (doc_id);    -- 문서 삭제·수정 때 그 문서의 행을 찾는다

CREATE TABLE bm25_doclen (
    doc_id bigint PRIMARY KEY,
    len    int    NOT NULL                       -- 그 문서의 tf 합 (조사·어미가 빠진 형태소 수)
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

-- ── 검색 함수 ────────────────────────────────────────────────────────
--   질의도 문서와 똑같이 to_tsvector('korean', ...) 로 형태소 분석한다
--   역색인에서 질의어가 있는 행만 읽으므로 전체 문서를 훑지 않는다
CREATE FUNCTION bm25_search(q text, k int DEFAULT 10,
                            k1 float8 DEFAULT 1.2, b float8 DEFAULT 0.75)
RETURNS TABLE (doc_id bigint, score float8)
LANGUAGE sql STABLE AS $$
  WITH qt AS (SELECT DISTINCT lexeme AS term FROM unnest(to_tsvector('korean', q))),
       s  AS (SELECT total_len::float8 / greatest(n_docs, 1) AS avgdl FROM bm25_stats)
  SELECT t.doc_id,
         sum(i.idf * t.tf * (k1 + 1) / (t.tf + k1 * (1 - b + b * dl.len / s.avgdl))) AS score
  FROM qt
  JOIN bm25_tf     t  ON t.term = qt.term
  JOIN bm25_idf    i  ON i.term = qt.term
  JOIN bm25_doclen dl ON dl.doc_id = t.doc_id
  CROSS JOIN s
  GROUP BY t.doc_id
  ORDER BY score DESC
  LIMIT k
$$;

-- ── IDF 계산 ────────────────────────────────────────────────────────
-- 전체 재계산 : N 이 바뀌면 원칙적으로 모든 단어의 IDF 가 바뀐다
CREATE FUNCTION bm25_refresh_idf() RETURNS void
LANGUAGE sql AS $$
  TRUNCATE bm25_idf;
  INSERT INTO bm25_idf (term, idf)
  SELECT d.term, ln(1 + (s.n_docs - d.df + 0.5) / (d.df + 0.5))
  FROM bm25_df d, bm25_stats s;
$$;

-- 부분 재계산 : 트리거가 건드린 단어만 최신 N 으로
--   (임시 테이블을 참조하므로 plpgsql. sql 함수는 만들 때 테이블 존재를 검사한다)
CREATE FUNCTION bm25_touch_idf() RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
  INSERT INTO bm25_idf (term, idf)
  SELECT d.term, ln(1 + (s.n_docs - d.df + 0.5) / (d.df + 0.5))
  FROM bm25_df d, bm25_stats s
  WHERE d.term IN (SELECT t.lexeme FROM _bm25_new n, unnest(n.tsv) t
                   UNION
                   SELECT t.lexeme FROM _bm25_old o, unnest(o.tsv) t)
  ON CONFLICT (term) DO UPDATE SET idf = EXCLUDED.idf;
END $$;

-- ── 문서 집합을 역색인에서 빼기 / 더하기 ─────────────────────────────
--   빼거나 더할 문서는 트리거가 임시 테이블 _bm25_old / _bm25_new 에 넣어 넘긴다
CREATE FUNCTION bm25_remove_docs() RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
  -- 그 문서들의 tf 행 삭제 → 삭제된 단어별로 df 감소
  WITH del AS (
    DELETE FROM bm25_tf t USING _bm25_old o WHERE t.doc_id = o.id RETURNING t.term
  ), cnt AS (
    SELECT term, count(*) AS c FROM del GROUP BY term
  )
  UPDATE bm25_df d SET df = d.df - cnt.c FROM cnt WHERE d.term = cnt.term;

  -- 더 이상 어떤 문서에도 없는 단어는 사전에서 제거
  DELETE FROM bm25_idf i USING bm25_df d WHERE i.term = d.term AND d.df <= 0;
  DELETE FROM bm25_df WHERE df <= 0;

  -- 문서 길이 · 전역 통계
  WITH del AS (
    DELETE FROM bm25_doclen l USING _bm25_old o WHERE l.doc_id = o.id RETURNING l.len
  )
  UPDATE bm25_stats
     SET n_docs    = n_docs    - (SELECT count(*) FROM del),
         total_len = total_len - (SELECT coalesce(sum(len), 0) FROM del);
END $$;

CREATE FUNCTION bm25_add_docs() RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
  INSERT INTO bm25_tf (term, doc_id, tf)
  SELECT t.lexeme, n.id, cardinality(t.positions)
  FROM _bm25_new n, unnest(n.tsv) t;

  INSERT INTO bm25_df (term, df)
  SELECT t.lexeme, count(*)
  FROM _bm25_new n, unnest(n.tsv) t
  GROUP BY t.lexeme
  ON CONFLICT (term) DO UPDATE SET df = bm25_df.df + EXCLUDED.df;

  WITH ins AS (
    INSERT INTO bm25_doclen (doc_id, len)
    SELECT n.id, coalesce((SELECT sum(cardinality(t.positions)) FROM unnest(n.tsv) t), 0)
    FROM _bm25_new n
    RETURNING len
  )
  UPDATE bm25_stats
     SET n_docs    = n_docs    + (SELECT count(*) FROM ins),
         total_len = total_len + (SELECT coalesce(sum(len), 0) FROM ins);
END $$;

-- ── 트리거 함수 ──────────────────────────────────────────────────────
-- INSERT · UPDATE · DELETE 공용. 문장(statement) 단위로 한 번, 바뀐 행 전체를 전이 테이블로 받는다
--   전이 테이블(new_rows / old_rows)은 트리거 함수 안에서만 보이므로 세션 임시 테이블로 옮겨 하위 함수와 공유
CREATE FUNCTION bm25_sync() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  -- embedding 만 바꾼 UPDATE(적재 노트북의 임베딩 채우기)는 역색인과 무관 → 바로 끝
  --   IF 두 단계로 나눈다. 한 조건식으로 쓰면 INSERT 때도 old_rows 를 참조해 오류가 난다
  IF TG_OP = 'UPDATE' THEN
    IF NOT EXISTS (SELECT 1 FROM old_rows o JOIN new_rows n USING (id)
                   WHERE o.tsv IS DISTINCT FROM n.tsv) THEN
      RETURN NULL;
    END IF;
  END IF;

  IF to_regclass('pg_temp._bm25_old') IS NULL THEN
    CREATE TEMP TABLE _bm25_old (id bigint, tsv tsvector);
    CREATE TEMP TABLE _bm25_new (id bigint, tsv tsvector);
  END IF;
  TRUNCATE _bm25_old, _bm25_new;

  IF TG_OP = 'INSERT' THEN
    INSERT INTO _bm25_new SELECT id, tsv FROM new_rows;
  ELSIF TG_OP = 'DELETE' THEN
    INSERT INTO _bm25_old SELECT id, tsv FROM old_rows;
  ELSE   -- UPDATE : tsv 가 실제로 바뀐 행만 "옛 문서 빼기 + 새 문서 더하기"
    INSERT INTO _bm25_old SELECT o.id, o.tsv FROM old_rows o JOIN new_rows n USING (id)
                          WHERE o.tsv IS DISTINCT FROM n.tsv;
    INSERT INTO _bm25_new SELECT n.id, n.tsv FROM old_rows o JOIN new_rows n USING (id)
                          WHERE o.tsv IS DISTINCT FROM n.tsv;
  END IF;

  PERFORM bm25_remove_docs();
  PERFORM bm25_add_docs();
  PERFORM bm25_touch_idf();
  RETURN NULL;
END $$;

-- TRUNCATE 는 행 트리거·DELETE 트리거를 부르지 않는다 → 역색인도 통째로 비운다
CREATE FUNCTION bm25_truncate() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  TRUNCATE bm25_tf, bm25_doclen, bm25_df, bm25_idf;
  UPDATE bm25_stats SET n_docs = 0, total_len = 0;
  RETURN NULL;
END $$;

-- ── 트리거 4개 ───────────────────────────────────────────────────────
--   전이 테이블은 이벤트 하나짜리 트리거에만 붙는다(PG 제약) → INSERT · UPDATE · DELETE 를 나눈다
CREATE TRIGGER bm25_sync_ins AFTER INSERT ON docs
  REFERENCING NEW TABLE AS new_rows
  FOR EACH STATEMENT EXECUTE FUNCTION bm25_sync();

CREATE TRIGGER bm25_sync_upd AFTER UPDATE ON docs
  REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
  FOR EACH STATEMENT EXECUTE FUNCTION bm25_sync();

CREATE TRIGGER bm25_sync_del AFTER DELETE ON docs
  REFERENCING OLD TABLE AS old_rows
  FOR EACH STATEMENT EXECUTE FUNCTION bm25_sync();

CREATE TRIGGER bm25_sync_trunc AFTER TRUNCATE ON docs
  FOR EACH STATEMENT EXECUTE FUNCTION bm25_truncate();

-- ── 전체 재구축 (트리거를 꺼 둔 사이 생긴 불일치 복구용) ─────────────
CREATE FUNCTION bm25_rebuild() RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
  TRUNCATE bm25_tf, bm25_doclen, bm25_df, bm25_idf;
  INSERT INTO bm25_tf (term, doc_id, tf)
    SELECT t.lexeme, d.id, cardinality(t.positions) FROM docs d, unnest(d.tsv) t;
  INSERT INTO bm25_doclen (doc_id, len)
    SELECT d.id, coalesce((SELECT sum(cardinality(t.positions)) FROM unnest(d.tsv) t), 0) FROM docs d;
  UPDATE bm25_stats SET n_docs = (SELECT count(*) FROM bm25_doclen),
                        total_len = (SELECT coalesce(sum(len), 0) FROM bm25_doclen);
  INSERT INTO bm25_df (term, df) SELECT term, count(*) FROM bm25_tf GROUP BY term;
  PERFORM bm25_refresh_idf();
END $$;
