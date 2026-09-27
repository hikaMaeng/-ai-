-- ==================================================================
-- 실습 7-2. 역색인 일괄 구축 + IDF 테이블
--   이미 적재된 docs 전체로부터 한 번에 채운다 (집합 연산 몇 번이면 끝)
-- ==================================================================

-- [1] TF : tsvector → (term, doc_id, tf)
TRUNCATE bm25_tf;
INSERT INTO bm25_tf (term, doc_id, tf)
SELECT t.lexeme, d.id, cardinality(t.positions)
FROM docs d, unnest(d.tsv) t;

-- [2] 문서 길이 |D| = 그 문서의 tf 합 (조사·어미가 빠진 "의미 형태소" 개수)
TRUNCATE bm25_doclen;
INSERT INTO bm25_doclen (doc_id, len)
SELECT d.id, coalesce((SELECT sum(cardinality(t.positions)) FROM unnest(d.tsv) t), 0)
FROM docs d;

-- [3] 전역 통계 N, 전체 길이 (avgdl = total_len / N)
UPDATE bm25_stats SET n_docs = (SELECT count(*) FROM bm25_doclen),
                      total_len = (SELECT coalesce(sum(len), 0) FROM bm25_doclen);

-- [4] DF : 단어별 등장 문서 수
TRUNCATE bm25_df;
INSERT INTO bm25_df (term, df)
SELECT term, count(*) FROM bm25_tf GROUP BY term;

-- [5] IDF : N 과 df 로 계산
--     N 이 바뀌면 모든 단어의 IDF 가 조금씩 바뀐다 → 전체 재계산 함수로 만들어 둔다
CREATE OR REPLACE FUNCTION bm25_refresh_idf() RETURNS void
LANGUAGE sql AS $$
  TRUNCATE bm25_idf;
  INSERT INTO bm25_idf (term, idf)
  SELECT d.term, ln(1 + (s.n_docs - d.df + 0.5) / (d.df + 0.5))
  FROM bm25_df d, bm25_stats s;
$$;
SELECT bm25_refresh_idf();

ANALYZE bm25_tf; ANALYZE bm25_doclen; ANALYZE bm25_df; ANALYZE bm25_idf;

-- [6] 확인
SELECT n_docs AS "N", total_len, round(total_len::numeric / n_docs, 1) AS avgdl FROM bm25_stats;
SELECT count(*) AS 역색인_행수 FROM bm25_tf;
SELECT count(*) AS 고유_단어수 FROM bm25_df;

-- 가장 흔한 단어 = IDF 최저 (슬라이드: "은·는·이·가·이다는 거의 0점")
SELECT d.term, d.df, round(i.idf::numeric, 3) AS idf
FROM bm25_df d JOIN bm25_idf i USING (term) ORDER BY d.df DESC LIMIT 15;

-- 희귀 단어 = IDF 최고
SELECT d.term, d.df, round(i.idf::numeric, 3) AS idf
FROM bm25_df d JOIN bm25_idf i USING (term)
WHERE d.term IN ('말라카이트', '반도체', '에디션', '트랜스포머', '사용', '것')
ORDER BY idf DESC;

-- df 구간별 단어 수 : 대부분의 단어는 1~2개 문서에만 나온다 (지프의 법칙)
SELECT CASE WHEN df = 1 THEN '1' WHEN df <= 5 THEN '2~5' WHEN df <= 50 THEN '6~50'
            WHEN df <= 500 THEN '51~500' ELSE '500+' END AS df_구간,
       count(*) AS 단어수
FROM bm25_df GROUP BY 1 ORDER BY min(df);
