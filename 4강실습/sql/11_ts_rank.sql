-- ==================================================================
-- 실습 6-1. ts_rank : 가중치 × 빈도
-- ==================================================================

-- [1] 기본 사용법: 매칭(@@)으로 거르고 ts_rank 로 정렬
SELECT id, title, ts_rank(tsv, q) AS rank
FROM docs, plainto_tsquery('korean', '반도체 수출') q
WHERE tsv @@ q
ORDER BY rank DESC
LIMIT 10;

-- [2] 구역 가중치의 효과
--     기본 가중치 배열은 {D, C, B, A} = {0.1, 0.2, 0.4, 1.0}
--     제목(A)에 등장한 문서가 본문(D)에만 등장한 문서보다 위로 온다
SELECT id, title,
       ts_rank(tsv, q)                          AS 기본,
       ts_rank('{0.1, 0.2, 0.4, 0.1}', tsv, q)  AS 제목가중치_끔,
       ts_rank('{1.0, 0.2, 0.4, 1.0}', tsv, q)  AS 본문도_1점
FROM docs, plainto_tsquery('korean', '삼성전자') q
WHERE tsv @@ q
ORDER BY 기본 DESC
LIMIT 10;
-- 질문: '제목가중치_끔' 컬럼으로 정렬하면 순위가 어떻게 바뀌는가?

-- [3] 반복 횟수에 대한 반응 (슬라이드: "saturation 이 없어 선형" 을 직접 검증)
--     같은 단어를 n번 반복한 문서의 점수를 비교한다
SELECT n,
       ts_rank(t, q)    AS ts_rank,
       ts_rank_cd(t, q) AS ts_rank_cd
FROM generate_series(1, 10) n,
     LATERAL (SELECT to_tsvector('korean', repeat('고양이 ', n) || '사료 추천') AS t,
                     plainto_tsquery('korean', '고양이') AS q) x;
-- 관찰: ts_rank 는 0.061 → 0.076 → 0.083 ... 0.094 로 증가폭이 줄어든다(포화).
--       PG 소스(tsrank.c calc_rank_or)는 j번째 등장에 w/(j+1)^2 를 더하고 π²/6 으로 나눈다.
--       반면 ts_rank_cd 는 0.1, 0.2, 0.3 ... 으로 선형 증가한다.

-- [4] AND 질의의 함정: 문장형 질문은 거의 매칭되지 않는다
--     plainto_tsquery 는 모든 형태소를 & 로 묶는다
SELECT plainto_tsquery('korean', '말라카이트에서 나온 색깔을 사용한 에디션은?');

SELECT count(*) AS and_매칭
FROM docs WHERE tsv @@ plainto_tsquery('korean', '말라카이트에서 나온 색깔을 사용한 에디션은?');

-- [5] OR 질의로 바꾸면 후보는 많아지고, 순위 함수의 품질이 중요해진다
--     & 를 | 로 바꿔서 다시 tsquery 로 캐스팅
SELECT replace(plainto_tsquery('korean', '말라카이트에서 나온 색깔을 사용한 에디션은?')::text, '&', '|')::tsquery;

WITH q AS (
  SELECT replace(plainto_tsquery('korean', '말라카이트에서 나온 색깔을 사용한 에디션은?')::text, '&', '|')::tsquery AS q
)
SELECT d.id, d.title, ts_rank(d.tsv, q.q) AS rank
FROM docs d, q
WHERE d.tsv @@ q.q
ORDER BY rank DESC
LIMIT 10;
-- 정답 지문은 'BMW 코리아 25주년 에디션' 기사(2위).
-- 1위는 '에디션'이 제목(A)에 들어간 다른 기사다. '말라카이트'는 코퍼스 전체에서 딱 1건에만 나오는
-- 희귀어인데 ts_rank 는 희귀도(IDF)를 모른다 → 7단계 BM25 에서 해결

-- [6] 편의 함수: 문장을 OR tsquery 로 (이후 실습에서 계속 사용)
CREATE OR REPLACE FUNCTION or_query(q text) RETURNS tsquery
LANGUAGE sql IMMUTABLE AS $$
  SELECT replace(plainto_tsquery('korean', q)::text, '&', '|')::tsquery
$$;

SELECT or_query('고양이 사료를 추천해 주세요');
