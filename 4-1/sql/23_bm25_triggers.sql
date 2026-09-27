-- ==================================================================
-- 실습 7-4. 트리거로 역색인 동적 갱신
--
--   docs 에 INSERT / UPDATE / DELETE 가 일어나면 bm25_* 테이블을 자동으로 맞춘다.
--
--   설계 포인트
--   1) 문장(statement) 단위 트리거 + 전이 테이블(transition table)
--      행마다 트리거를 돌리면 1만 건 INSERT 에 1만 번 실행된다.
--      REFERENCING NEW TABLE AS new_rows 를 쓰면 문장당 1번, 바뀐 행 전체를 집합으로 받는다.
--      (PG 제약: 전이 테이블은 이벤트 하나짜리 트리거에만 붙는다 → INSERT/UPDATE/DELETE 3개로 나눈다)
--   2) UPDATE 는 "옛 문서 빼기 + 새 문서 더하기"
--      단, tsv 가 그대로인 UPDATE(예: embedding 만 수정)는 건너뛴다.
--   3) IDF 는 "건드린 단어"만 즉시 다시 계산한다.
--      N 이 바뀌면 원칙적으로 모든 단어의 IDF 가 바뀌지만, 문서 몇 건으로는 차이가 미미하다.
--      → 평소엔 부분 갱신, 주기적으로 bm25_refresh_idf() 로 전체 재계산 (Lucene 세그먼트 병합과 같은 발상)
-- ==================================================================

-- [1] 문서 집합을 역색인에서 빼는 함수 / 더하는 함수
--     트리거 함수 안에서 전이 테이블을 임시 테이블로 넘겨받아 재사용한다
CREATE OR REPLACE FUNCTION bm25_remove_docs() RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
  -- 해당 문서들의 tf 행 삭제 → 삭제된 단어별로 df 감소
  WITH del AS (
    DELETE FROM bm25_tf t USING _bm25_old o WHERE t.doc_id = o.id RETURNING t.term
  ), cnt AS (
    SELECT term, count(*) AS c FROM del GROUP BY term
  )
  UPDATE bm25_df d SET df = d.df - cnt.c FROM cnt WHERE d.term = cnt.term;

  -- 더 이상 어떤 문서에도 없는 단어는 사전에서 제거
  DELETE FROM bm25_idf i USING bm25_df d WHERE i.term = d.term AND d.df <= 0;
  DELETE FROM bm25_df WHERE df <= 0;

  -- 문서 길이·전역 통계
  WITH del AS (
    DELETE FROM bm25_doclen l USING _bm25_old o WHERE l.doc_id = o.id RETURNING l.len
  )
  UPDATE bm25_stats
     SET n_docs    = n_docs    - (SELECT count(*) FROM del),
         total_len = total_len - (SELECT coalesce(sum(len), 0) FROM del);
END $$;

CREATE OR REPLACE FUNCTION bm25_add_docs() RETURNS void
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

-- 건드린 단어들의 IDF 만 최신 N 으로 다시 계산
--   (임시 테이블을 참조하므로 LANGUAGE sql 이 아니라 plpgsql : sql 함수는 생성 시점에 테이블 존재를 검사한다)
CREATE OR REPLACE FUNCTION bm25_touch_idf() RETURNS void
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

-- [2] 트리거 함수 : TG_OP 에 따라 전이 테이블을 _bm25_old / _bm25_new 임시 테이블로 옮긴다
--     전이 테이블(new_rows/old_rows)은 트리거 함수 안에서만 보이므로, 하위 함수들과 공유하려고
--     세션 임시 테이블에 복사한다
CREATE OR REPLACE FUNCTION bm25_sync() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF to_regclass('pg_temp._bm25_old') IS NULL THEN
    CREATE TEMP TABLE _bm25_old (id bigint, tsv tsvector);
    CREATE TEMP TABLE _bm25_new (id bigint, tsv tsvector);
  END IF;
  TRUNCATE _bm25_old, _bm25_new;

  IF TG_OP = 'INSERT' THEN
    INSERT INTO _bm25_new SELECT id, tsv FROM new_rows;
  ELSIF TG_OP = 'DELETE' THEN
    INSERT INTO _bm25_old SELECT id, tsv FROM old_rows;
  ELSIF TG_OP = 'UPDATE' THEN
    -- tsv 가 실제로 바뀐 행만 (제목/본문 수정). embedding 만 바꾼 UPDATE 는 무시
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

-- [3] 트리거 3개 (이벤트별로 하나씩)
DROP TRIGGER IF EXISTS bm25_sync_ins ON docs;
DROP TRIGGER IF EXISTS bm25_sync_upd ON docs;
DROP TRIGGER IF EXISTS bm25_sync_del ON docs;

CREATE TRIGGER bm25_sync_ins AFTER INSERT ON docs
  REFERENCING NEW TABLE AS new_rows
  FOR EACH STATEMENT EXECUTE FUNCTION bm25_sync();

CREATE TRIGGER bm25_sync_upd AFTER UPDATE ON docs
  REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
  FOR EACH STATEMENT EXECUTE FUNCTION bm25_sync();

CREATE TRIGGER bm25_sync_del AFTER DELETE ON docs
  REFERENCING OLD TABLE AS old_rows
  FOR EACH STATEMENT EXECUTE FUNCTION bm25_sync();

-- [4] 전체 재구축 함수 (트리거가 꺼져 있던 동안 쌓인 불일치 복구용, 21번 파일의 내용을 한 번에)
CREATE OR REPLACE FUNCTION bm25_rebuild() RETURNS void
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

SELECT tgname, tgenabled FROM pg_trigger WHERE tgrelid = 'docs'::regclass AND NOT tgisinternal;
