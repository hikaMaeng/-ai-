-- ==================================================================
-- 실습 6-3. 길이 정규화 옵션 (ts_rank 의 세 번째 인자)
--   0  : 정규화 없음 (기본)
--   1  : rank / (1 + log(문서 길이))   → 완만한 로그 감쇄
--   2  : rank / 문서 길이               → 급격한 역수 감쇄
--   4  : rank / 커버 간 평균 거리 (ts_rank_cd 전용)
--   8  : rank / 고유 단어 수
--   16 : rank / (1 + log(고유 단어 수))
--   32 : rank / (rank + 1)              → 0~1 로 눌러 담기
--   여러 개를 | 로 조합할 수 있다 (예: 1|32)
-- ==================================================================

-- [1] 짧은 문서 vs 긴 문서
WITH t(name, body) AS (VALUES
  ('짧음(5단어)',   '서울 아파트 가격 상승 전망'),
  ('중간(40단어)',  '서울 아파트 가격 ' || repeat('경제 동향 분석 ', 12) || '상승'),
  ('김(300단어)',   '서울 아파트 가격 ' || repeat('경제 동향 분석 ', 100) || '상승')
)
SELECT name,
       length(to_tsvector('korean', body))             AS 고유단어,
       round(ts_rank(to_tsvector('korean', body), q, 0)::numeric, 4) AS opt0,
       round(ts_rank(to_tsvector('korean', body), q, 1)::numeric, 4) AS opt1_log,
       round(ts_rank(to_tsvector('korean', body), q, 2)::numeric, 4) AS opt2_len,
       round(ts_rank(to_tsvector('korean', body), q, 32)::numeric, 4) AS opt32
FROM t, or_query('서울 아파트 가격') q;
-- 관찰: opt1 은 긴 문서를 살짝만 깎고, opt2 는 길이에 반비례해서 크게 깎는다

-- [2] 실제 데이터에서 옵션별 상위 결과의 평균 길이
--     opt2 는 짧은 문서를 끌어올린다 → 제목·FAQ 처럼 짧고 정확한 매칭에 유리
WITH q AS (SELECT or_query('코로나 백신 접종') AS q)
SELECT opt,
       round(avg(length(content))) AS 상위10_평균글자수,
       string_agg(id::text, ',' ORDER BY rn) AS 상위10_id
FROM (
  SELECT o.opt, d.id, d.content,
         row_number() OVER (PARTITION BY o.opt ORDER BY ts_rank(d.tsv, q.q, o.opt) DESC) AS rn
  FROM docs d, q, (VALUES (0), (1), (2), (8)) o(opt)
  WHERE d.tsv @@ q.q
) x
WHERE rn <= 10
GROUP BY opt ORDER BY opt;

-- [3] PG 정규화의 한계
--     "길이"는 그 문서 하나의 길이일 뿐, 코퍼스 평균 길이(avgdl)를 모른다.
--     따라서 "평균보다 긴가 짧은가"로 보간하는 BM25 의 b 파라미터를 흉내 낼 수 없다.
SELECT round(avg(length(tsv)), 1) AS 평균_고유단어수,
       min(length(tsv)), max(length(tsv))
FROM docs;
