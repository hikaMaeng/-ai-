-- ==================================================================
-- 실습 7-3. BM25 점수 계산 → 검색 함수
-- ==================================================================

-- [1] 공식을 항 단위로 쪼개 보기 : 질문 하나, 정답 문서 하나
--     tf_part = tf·(k1+1) / (tf + k1·(1 − b + b·len/avgdl))     k1 = 1.2, b = 0.75
WITH params AS (SELECT 1.2::float8 AS k1, 0.75::float8 AS b),
     s AS (SELECT n_docs, total_len::float8 / n_docs AS avgdl FROM bm25_stats),
     q AS (SELECT DISTINCT lexeme AS term
           FROM unnest(to_tsvector('korean', '말라카이트에서 나온 색깔을 사용한 에디션은?')))
SELECT q.term,
       df.df,
       round(i.idf::numeric, 3)                              AS idf,
       coalesce(t.tf, 0)                                     AS tf,
       dl.len, round(s.avgdl::numeric, 1)                    AS avgdl,
       round((coalesce(t.tf, 0) * (p.k1 + 1)
             / (coalesce(t.tf, 0) + p.k1 * (1 - p.b + p.b * dl.len / s.avgdl)))::numeric, 3) AS tf_part,
       round((i.idf * coalesce(t.tf, 0) * (p.k1 + 1)
             / (coalesce(t.tf, 0) + p.k1 * (1 - p.b + p.b * dl.len / s.avgdl)))::numeric, 3) AS score
FROM q
CROSS JOIN params p CROSS JOIN s
JOIN bm25_doclen dl ON dl.doc_id = 1                    -- 정답 지문 (BMW 25주년 에디션)
LEFT JOIN bm25_df  df ON df.term = q.term
LEFT JOIN bm25_idf i  ON i.term  = q.term
LEFT JOIN bm25_tf  t  ON t.term  = q.term AND t.doc_id = dl.doc_id
ORDER BY score DESC NULLS LAST;
-- 관찰: 정답 지문의 점수는 희귀어 두 개가 거의 다 만든다
--       '말라카이트'(df=1, IDF 8.2)는 tf=3 인데도 12.96 점, '에디션'(df=16)은 tf=10 이지만 포화돼서 11.38 점
--       '사용'(df=906)이 이 지문에 있었다 해도 IDF 1.77 이라 기여는 3점 안쪽이다

-- [2] TF 포화 곡선 : k1 이 클수록 포화가 늦다 (TF 가 점수에 오래 기여한다)
--     |D| = avgdl 이라고 가정 → tf_part = tf·(k1+1)/(tf+k1)
SELECT tf,
       round(tf * 2.2 / (tf + 1.2), 3) AS "k1=1.2",
       round(tf * 3.0 / (tf + 2.0), 3) AS "k1=2.0",
       round(tf * 6.0 / (tf + 5.0), 3) AS "k1=5.0",
       tf                              AS "선형(포화 없음)"
FROM (VALUES (1), (2), (3), (5), (10), (20), (50)) v(tf);
-- 상한은 k1+1. k1=1.2 는 tf 5 정도에서 이미 상한의 80% 에 도달한다

-- [3] 길이 정규화 b : 평균보다 긴 문서는 감점, 짧은 문서는 가점
SELECT len_ratio AS "len/avgdl",
       round(3 * 2.2 / (3 + 1.2 * (1 - 0.0  + 0.0  * len_ratio)), 3) AS "b=0",
       round(3 * 2.2 / (3 + 1.2 * (1 - 0.75 + 0.75 * len_ratio)), 3) AS "b=0.75",
       round(3 * 2.2 / (3 + 1.2 * (1 - 1.0  + 1.0  * len_ratio)), 3) AS "b=1"
FROM (VALUES (0.25), (0.5), (1.0), (2.0), (4.0)) v(len_ratio);   -- tf = 3 고정

-- [4] 검색 함수
--     질의도 문서와 똑같이 to_tsvector('korean', ...) 로 형태소 분석한다
--     역색인에서 질의어가 있는 행만 읽으므로 전체 문서를 훑지 않는다
CREATE OR REPLACE FUNCTION bm25_search(q text, k int DEFAULT 10,
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

-- [5] ts_rank 에서 2위였던 정답이 1위로 올라오는지 확인
SELECT r.doc_id, round(r.score::numeric, 3) AS score, d.title
FROM bm25_search('말라카이트에서 나온 색깔을 사용한 에디션은?') r
JOIN docs d ON d.id = r.doc_id
ORDER BY r.score DESC;

-- [6] 파라미터 바꿔 보기 : b = 0 (길이 정규화 끔) vs b = 1 (최대)
SELECT 'b=0' AS 설정, string_agg(doc_id::text, ', ' ORDER BY score DESC) AS 상위5
FROM bm25_search('서울 아파트 가격 상승', 5, 1.2, 0)
UNION ALL
SELECT 'b=0.75', string_agg(doc_id::text, ', ' ORDER BY score DESC)
FROM bm25_search('서울 아파트 가격 상승', 5)
UNION ALL
SELECT 'b=1', string_agg(doc_id::text, ', ' ORDER BY score DESC)
FROM bm25_search('서울 아파트 가격 상승', 5, 1.2, 1);

-- [7] 실행 계획 : 역색인 PK (term, doc_id) 를 타는지
EXPLAIN ANALYZE SELECT * FROM bm25_search('반도체 공장 투자 발표');
