-- ==================================================================
-- 실습 7-5. 트리거 동작 검증
--   새 문서를 넣고, 고치고, 지우면서 통계가 따라오는지 본다.
--   마지막에 "트리거로 유지한 값 == 처음부터 다시 계산한 값" 인지 대조한다.
-- ==================================================================

-- [0] 현재 상태
SELECT n_docs, total_len FROM bm25_stats;
SELECT * FROM bm25_df WHERE term IN ('양자컴퓨터', '큐비트', '말라카이트');

-- [1] INSERT : 코퍼스에 없던 단어('큐비트')가 들어간 문서 2건
INSERT INTO docs (doc_key, title, content, source) VALUES
  ('test-1', '양자컴퓨터 상용화 경쟁', '국내 연구진이 큐비트 100개 규모의 양자컴퓨터 시제품을 공개했다. 큐비트 오류율을 크게 낮췄다.', 'test'),
  ('test-2', '말라카이트 원석 전시회', '녹색 광물 말라카이트 원석을 모은 전시회가 열린다.', 'test');

SELECT n_docs, total_len FROM bm25_stats;                               -- N 이 2 늘었다
SELECT d.term, d.df, round(i.idf::numeric, 3) AS idf
FROM bm25_df d JOIN bm25_idf i USING (term)
WHERE term IN ('양자컴퓨터', '큐비트', '말라카이트');                    -- 말라카이트 df 1 → 2, IDF 하락

-- 새 문서가 바로 검색된다
SELECT r.*, d.title FROM bm25_search('큐비트 양자컴퓨터') r JOIN docs d ON d.id = r.doc_id;
SELECT r.*, d.title FROM bm25_search('말라카이트에서 나온 색깔을 사용한 에디션은?', 3) r JOIN docs d ON d.id = r.doc_id;

-- [2] UPDATE (본문 변경) : '큐비트' 를 빼면 df 가 줄어야 한다
UPDATE docs SET content = '국내 연구진이 양자컴퓨터 시제품을 공개했다.' WHERE doc_key = 'test-1';
SELECT * FROM bm25_df WHERE term = '큐비트';                            -- 0건 → 사전에서 사라짐

-- [3] UPDATE (tsv 무관한 컬럼) : 역색인은 건드리지 않아야 한다
UPDATE docs SET source = 'test-updated' WHERE doc_key LIKE 'test-%';
SELECT n_docs, total_len FROM bm25_stats;                               -- 변화 없음

-- [4] DELETE
DELETE FROM docs WHERE doc_key LIKE 'test-%';
SELECT n_docs, total_len FROM bm25_stats;                               -- 원래 값으로
SELECT * FROM bm25_df WHERE term IN ('양자컴퓨터', '말라카이트');         -- 말라카이트 df 다시 1

-- [5] 정합성 검사 : 트리거로 유지된 값 vs 처음부터 다시 계산한 값
--     모든 행이 0 이어야 정상
SELECT 'tf 불일치' AS 검사, count(*) FROM (
  (SELECT term, doc_id, tf FROM bm25_tf
   EXCEPT SELECT t.lexeme, d.id, cardinality(t.positions) FROM docs d, unnest(d.tsv) t)
  UNION ALL
  (SELECT t.lexeme, d.id, cardinality(t.positions) FROM docs d, unnest(d.tsv) t
   EXCEPT SELECT term, doc_id, tf FROM bm25_tf)
) x
UNION ALL
SELECT 'df 불일치', count(*) FROM (
  (SELECT term, df FROM bm25_df EXCEPT SELECT term, count(*)::int FROM bm25_tf GROUP BY term)
  UNION ALL
  (SELECT term, count(*)::int FROM bm25_tf GROUP BY term EXCEPT SELECT term, df FROM bm25_df)
) x
UNION ALL
SELECT 'N 불일치', abs((SELECT n_docs FROM bm25_stats) - (SELECT count(*) FROM docs))
UNION ALL
SELECT 'IDF 오차>1e-9', count(*)
FROM bm25_idf i JOIN bm25_df d USING (term), bm25_stats s
WHERE abs(i.idf - ln(1 + (s.n_docs - d.df + 0.5) / (d.df + 0.5))) > 1e-9;
-- 결과 : tf / df / N 은 0 (트리거가 정확히 유지)
--        IDF 오차는 몇 건(약 7건) 나온다. 버그가 아니라 설계다.
--        [2]의 UPDATE 때 빠진 단어('연구진','오류율' 등, 다른 문서에도 있는 단어)는 그때의 N=5311 로
--        IDF 가 계산됐고, [4]의 DELETE 는 수정 후 본문의 단어만 건드리므로 그 단어들은 다시 계산되지 않았다.
--        트리거는 "건드린 단어"만 갱신한다 → 오차가 쌓이면 전체 재계산
SELECT bm25_refresh_idf();
-- 다시 [5] 를 실행하면 IDF 오차도 0

-- [6] 대량 적재도 트리거가 처리한다 (문장 단위 트리거라 빠르다)
--     loader 를 다시 실행하면 DELETE(5,309행) → COPY(5,309행) 가 각각 트리거 1회로 처리된다
--       docker compose run --rm loader --no-embed     (텍스트만 재적재, 1분)
--     실행 후 아래로 확인 : N 이 docs 행 수와 같아야 한다
SELECT (SELECT n_docs FROM bm25_stats) AS bm25_N, (SELECT count(*) FROM docs) AS docs;
--     (재적재하면 임베딩이 지워지므로 25번 전에 docker compose run --rm loader --embed-only)
