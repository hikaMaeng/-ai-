-- ==================================================================
-- 실습 6-4. 검색 품질 측정 : Hit@10 / MRR
--   questions 의 각 질문에 대해 "정답 지문(doc_id)이 상위 10위 안에 들었나"를 센다
--   MRR = 정답 순위 역수의 평균 (1위면 1, 2위면 0.5, 10위 밖이면 0)
--   시간 절약을 위해 질문 500개(id % 11 = 0)만 샘플링한다
-- ==================================================================

-- [1] 평가용 샘플
DROP TABLE IF EXISTS eval_q;
CREATE TABLE eval_q AS
SELECT id, doc_id, question FROM questions WHERE id % 11 = 0 ORDER BY id LIMIT 500;

-- [2] 방법별 순위 계산 → 정답 순위만 뽑는 함수
--     method: 'and' | 'or_rank' | 'or_rank_cd' | 'or_rank_norm1' | 'or_rank_norm2'
CREATE OR REPLACE FUNCTION eval_tsrank(method text, k int DEFAULT 10)
RETURNS TABLE (method_name text, questions bigint, hit_at_k numeric, mrr numeric)
LANGUAGE sql AS $$
  WITH ranked AS (
    SELECT e.id AS qid, e.doc_id AS gold, r.doc_id, r.rn
    FROM eval_q e
    -- 질문의 tsquery 는 질문당 한 번만 만든다 (문서 행마다 형태소 분석을 반복하지 않도록)
    CROSS JOIN LATERAL (
      SELECT CASE WHEN method = 'and' THEN plainto_tsquery('korean', e.question)
                  ELSE or_query(e.question) END AS q
    ) qq
    CROSS JOIN LATERAL (
      SELECT d.id AS doc_id,
             row_number() OVER (ORDER BY
               CASE method
                 WHEN 'and'           THEN ts_rank(d.tsv, qq.q)
                 WHEN 'or_rank'       THEN ts_rank(d.tsv, qq.q)
                 WHEN 'or_rank_cd'    THEN ts_rank_cd(d.tsv, qq.q)
                 WHEN 'or_rank_norm1' THEN ts_rank(d.tsv, qq.q, 1)
                 WHEN 'or_rank_norm2' THEN ts_rank(d.tsv, qq.q, 2)
               END DESC) AS rn
      FROM docs d
      WHERE d.tsv @@ qq.q
      ORDER BY rn LIMIT k
    ) r
  )
  SELECT method,
         (SELECT count(*) FROM eval_q),
         round(count(*) FILTER (WHERE doc_id = gold)::numeric / (SELECT count(*) FROM eval_q), 3),
         round(sum(CASE WHEN doc_id = gold THEN 1.0 / rn ELSE 0 END) / (SELECT count(*) FROM eval_q), 3)
  FROM ranked
$$;

-- [3] 비교 : 결과를 eval_result 에 쌓아 두고 25번에서 BM25·벡터와 함께 본다
--     OR 질의는 흔한 형태소 때문에 문서 대부분이 매칭되어 질문당 ~0.1초 → 방법당 약 1분
DROP TABLE IF EXISTS eval_result;
CREATE TABLE eval_result (method_name text PRIMARY KEY, questions bigint, hit_at_k numeric, mrr numeric);

INSERT INTO eval_result SELECT * FROM eval_tsrank('and');          -- 1초
INSERT INTO eval_result SELECT * FROM eval_tsrank('or_rank');      -- 약 1분
INSERT INTO eval_result SELECT * FROM eval_tsrank('or_rank_cd');   -- 약 1분
SELECT * FROM eval_result;

-- [4] (선택, 시간 여유가 있을 때) 길이 정규화 옵션 비교 — 각 1분
-- INSERT INTO eval_result SELECT * FROM eval_tsrank('or_rank_norm1');
-- INSERT INTO eval_result SELECT * FROM eval_tsrank('or_rank_norm2');
-- SELECT * FROM eval_result;

-- 참고 결과 (validation 적재 기준, 동점 처리 순서에 따라 ±0.01)
--   and 0.048 / or_rank 0.484 / or_rank_cd 0.524 / norm1 0.484 / norm2 0.374
-- 생각해 볼 것
--   * AND 는 정확하지만 대부분의 질문에서 아무것도 못 찾는다 (재현율 붕괴)
--   * norm1 은 정답 순위를 198개 질문에서 바꿨지만 합계는 거의 같다.
--     지문 길이가 고른 코퍼스(고유 형태소 31~358개)에서는 로그 감쇄의 영향이 작다
--   * norm2 는 짧은 지문을 과하게 끌어올려 오히려 나빠진다 (슬라이드: 긴 청크 RAG 에는 opt1)
--   * OR + ts_rank 는 흔한 단어('하다','있다','년')가 많이 나온 문서에 끌려간다
--   * 이 점수표를 7단계 BM25 결과와 비교한다
