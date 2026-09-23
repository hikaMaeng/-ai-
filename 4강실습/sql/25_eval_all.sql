-- ==================================================================
-- 실습 7-6. 종합 비교 : ts_rank vs BM25 vs 벡터 vs 하이브리드(RRF)
--   14_eval_ts_rank.sql 의 eval_q(질문 500개)를 그대로 쓴다
--   벡터·하이브리드는 loader 의 임베딩 단계가 끝나야 한다 ('완료' 출력 확인)
-- ==================================================================

-- [1] 방법별로 정답 지문의 순위(1~10, 없으면 NULL)를 구하는 함수
CREATE OR REPLACE FUNCTION eval_hits(method text, k int DEFAULT 10)
RETURNS TABLE (qid bigint, gold_rank int)
LANGUAGE plpgsql AS $$
BEGIN
  IF method = 'bm25' THEN
    RETURN QUERY
    SELECT e.id, (SELECT r.rn::int FROM (
              SELECT b.doc_id, row_number() OVER (ORDER BY b.score DESC) AS rn
              FROM bm25_search(e.question, k) b) r WHERE r.doc_id = e.doc_id)
    FROM eval_q e;

  ELSIF method = 'vector' THEN
    -- 질문 임베딩과 지문 임베딩의 코사인 거리(<=>) 상위 k
    RETURN QUERY
    SELECT e.id, (SELECT r.rn::int FROM (
              SELECT d.id AS doc_id, row_number() OVER (ORDER BY d.embedding <=> q.embedding) AS rn
              FROM docs d ORDER BY d.embedding <=> q.embedding LIMIT k) r WHERE r.doc_id = e.doc_id)
    FROM eval_q e JOIN questions q ON q.id = e.id;

  ELSIF method = 'hybrid' THEN
    -- RRF(Reciprocal Rank Fusion) : 점수 대신 "순위"만 합친다. score = Σ 1/(60 + rank)
    -- BM25 점수와 코사인 거리는 척도가 달라 직접 더할 수 없기 때문
    RETURN QUERY
    SELECT e.id, (SELECT f.rn::int FROM (
              SELECT u.doc_id, row_number() OVER (ORDER BY sum(1.0 / (60 + u.rn)) DESC) AS rn
              FROM (
                SELECT b.doc_id, row_number() OVER (ORDER BY b.score DESC) AS rn
                FROM bm25_search(e.question, 50) b
                UNION ALL
                SELECT v.id, row_number() OVER (ORDER BY v.embedding <=> q.embedding)
                FROM (SELECT d.id, d.embedding FROM docs d ORDER BY d.embedding <=> q.embedding LIMIT 50) v
              ) u GROUP BY u.doc_id) f WHERE f.doc_id = e.doc_id AND f.rn <= k)
    FROM eval_q e JOIN questions q ON q.id = e.id;

  ELSE  -- ts_rank 계열은 14번의 eval_tsrank 로
    RAISE EXCEPTION 'unknown method %', method;
  END IF;
END $$;

CREATE OR REPLACE FUNCTION eval_summary(method text, k int DEFAULT 10)
RETURNS TABLE (method_name text, hit_at_k numeric, mrr numeric)
LANGUAGE sql AS $$
  SELECT method,
         round(avg(CASE WHEN gold_rank IS NOT NULL THEN 1 ELSE 0 END), 3),
         round(avg(coalesce(1.0 / gold_rank, 0)), 3)
  FROM eval_hits(method, k)
$$;

-- [2] 종합표 : 14번에서 쌓은 ts_rank 결과 + BM25 · 벡터 · 하이브리드 (약 40초)
INSERT INTO eval_result SELECT method_name, (SELECT count(*) FROM eval_q), hit_at_k, mrr FROM eval_summary('bm25')
  ON CONFLICT (method_name) DO UPDATE SET hit_at_k = EXCLUDED.hit_at_k, mrr = EXCLUDED.mrr;
INSERT INTO eval_result SELECT method_name, (SELECT count(*) FROM eval_q), hit_at_k, mrr FROM eval_summary('vector')
  ON CONFLICT (method_name) DO UPDATE SET hit_at_k = EXCLUDED.hit_at_k, mrr = EXCLUDED.mrr;
INSERT INTO eval_result SELECT method_name, (SELECT count(*) FROM eval_q), hit_at_k, mrr FROM eval_summary('hybrid')
  ON CONFLICT (method_name) DO UPDATE SET hit_at_k = EXCLUDED.hit_at_k, mrr = EXCLUDED.mrr;
SELECT method_name, hit_at_k, mrr FROM eval_result ORDER BY hit_at_k DESC;
-- 참고 결과 (validation 적재, 질문 500개, 동점 처리 순서에 따라 ±0.01)
--   method       hit@10  mrr
--   bm25         0.938   0.838   ← IDF 하나로 두 배 가까이
--   hybrid       0.896   0.523   ← 약한 벡터가 순위를 끌어내린다
--   or_rank_cd   0.524   0.329
--   or_rank      0.484   0.354
--   vector       0.376   0.204
--   and          0.048   0.044

-- [3] BM25 파라미터 튜닝 : k1, b 를 바꿔 가며 Hit@10
--     (bm25_search 의 기본값을 바꾸는 대신 직접 호출)
SELECT k1, b,
       round(avg(CASE WHEN EXISTS (SELECT 1 FROM bm25_search(e.question, 10, k1, b) r
                                   WHERE r.doc_id = e.doc_id) THEN 1 ELSE 0 END), 3) AS hit_at_10
FROM eval_q e,
     (VALUES (1.2, 0.0), (1.2, 0.75), (1.2, 1.0), (0.5, 0.75), (2.0, 0.75)) p(k1, b)
GROUP BY k1, b ORDER BY k1, b;

-- [4] 방법 간 실패 사례 비교 : BM25 는 맞히고 벡터는 틀린 질문 / 그 반대
WITH bm AS (SELECT * FROM eval_hits('bm25')),
     ve AS (SELECT * FROM eval_hits('vector'))
SELECT CASE WHEN bm.gold_rank IS NOT NULL AND ve.gold_rank IS NULL THEN 'BM25만 성공'
            WHEN bm.gold_rank IS NULL AND ve.gold_rank IS NOT NULL THEN '벡터만 성공' END AS 구분,
       q.question, bm.gold_rank AS bm25순위, ve.gold_rank AS 벡터순위
FROM bm JOIN ve USING (qid) JOIN questions q ON q.id = bm.qid
WHERE (bm.gold_rank IS NULL) <> (ve.gold_rank IS NULL)
ORDER BY 1, q.id
LIMIT 20;
-- 생각해 볼 것
--   * KLUE-MRC 질문은 지문의 고유명사·숫자를 그대로 쓰는 경우가 많아 키워드 검색에 유리한 데이터다
--   * [3] 에서 b=0 이 오히려 좋다 : 지문 길이가 고른 코퍼스에서는 길이 보간이 잡음이 된다
--   * 벡터 검색이 BM25 보다 낮은 이유 : 지문(~1천자)을 통째로 임베딩했다.
--     MiniLM 의 입력 한계는 128 토큰이라 지문 앞부분(대략 첫 2~3문장)만 벡터에 담긴다.
--     정답이 지문 뒤쪽에 있으면 벡터로는 찾을 수 없다 → 청킹의 중요성 (슬라이드 "청킹이 가장 중요")
--   * 하이브리드가 BM25 보다 낮다. RRF 는 두 목록을 동등하게 섞으므로 한쪽이 크게 약하면 손해다.
--     [4] 처럼 서로 다른 질문을 맞히는 건 사실이므로, 가중 RRF(bm25 쪽 가중치↑)나 청킹 후 재측정해 볼 것
