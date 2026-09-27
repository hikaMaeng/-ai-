-- ==================================================================
-- 실습 6-2. ts_rank_cd : 커버 밀집도(Cover Density)
--   질의어들이 가까이 모여 있을수록 점수가 높다
-- ==================================================================

-- [1] 같은 단어 구성, 다른 거리
--     A: 두 단어가 붙어 있음 / B: 사이에 단어 8개 / C: 사이에 단어 30개
WITH t(name, body) AS (VALUES
  ('A 붙어있음', '인공지능 모델 성능 평가 보고서'),
  ('B 떨어짐',   '인공지능 분야 최근 동향 정리 시장 규모 투자 현황 모델 비교'),
  ('C 멀리',     '인공지능 ' || repeat('기타 ', 30) || '모델')
)
SELECT name,
       ts_rank(to_tsvector('korean', body), q)    AS ts_rank,
       ts_rank_cd(to_tsvector('korean', body), q) AS ts_rank_cd
FROM t, to_tsquery('korean', '인공지능 & 모델') q;
-- 관찰: 둘 다 거리가 멀수록 점수가 떨어진다.
--       ts_rank 도 AND 질의(&)에서는 단어 간 거리를 반영한다 (tsrank.c calc_rank_and).
--       ts_rank_cd 는 '질의어를 모두 포함하는 최소 구간(커버)'의 길이에 반비례한다.
--       OR 질의(|)로 바꾸면 어떻게 될까? → 아래 [1-2]

-- [1-2] OR 질의에서는 둘 다 거리를 보지 않는다
WITH t(name, body) AS (VALUES
  ('A 붙어있음', '인공지능 모델 성능 평가 보고서'),
  ('C 멀리',     '인공지능 ' || repeat('기타 ', 30) || '모델')
)
SELECT name,
       ts_rank(to_tsvector('korean', body), q)    AS ts_rank,
       ts_rank_cd(to_tsvector('korean', body), q) AS ts_rank_cd
FROM t, to_tsquery('korean', '인공지능 | 모델') q;
-- 관찰: A 와 C 의 점수가 똑같다. OR 에서는 단어 하나만으로 커버가 성립하므로
--       밀집도 효과가 사라진다. 문장형 질문을 OR 로 풀면 cd 의 장점이 약해지는 이유

-- [2] 커버(cover) 여러 개: 질의어 묶음이 여러 번 등장하면 커버 점수가 합산된다
WITH t(name, body) AS (VALUES
  ('커버 1개', '인공지능 모델 기타 기타 기타 기타 기타 기타'),
  ('커버 2개', '인공지능 모델 기타 기타 기타 인공지능 모델 기타'),
  ('커버 3개', '인공지능 모델 기타 인공지능 모델 기타 인공지능 모델')
)
SELECT name, ts_rank_cd(to_tsvector('korean', body), to_tsquery('korean', '인공지능 & 모델')) AS cd
FROM t;

-- [3] 실제 데이터: ts_rank 와 ts_rank_cd 순위 비교
WITH q AS (SELECT or_query('반도체 공장 투자 발표') AS q),
r AS (
  SELECT d.id, d.title,
         rank() OVER (ORDER BY ts_rank(d.tsv, q.q) DESC)    AS rank_ts,
         rank() OVER (ORDER BY ts_rank_cd(d.tsv, q.q) DESC) AS rank_cd
  FROM docs d, q WHERE d.tsv @@ q.q
)
SELECT * FROM r WHERE rank_ts <= 10 OR rank_cd <= 10 ORDER BY rank_cd;

-- [4] ts_rank_cd 는 위치 정보가 필요하다. strip() 으로 위치를 지우면 0점
SELECT ts_rank_cd(strip(to_tsvector('korean', '인공지능 모델')), to_tsquery('korean', '인공지능 & 모델'));
