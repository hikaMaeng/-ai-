-- ==================================================================
-- 실습 6-0. 형태소 분석과 tsvector
--   pgAdmin 쿼리 도구에서 블록 단위로 드래그해 실행(F5)하거나
--   docker compose exec db psql -U lab -d lab -f /sql/10_tokenize.sql
-- ==================================================================

-- [1] mecab-ko 가 문장을 어떻게 쪼개는가 (품사 태그 포함)
SELECT word, type, basic FROM mecabko_analyze('아버지가방에들어가신다');
SELECT word, type, basic FROM mecabko_analyze('가셨습니다');          -- 슬라이드의 Mecab-ko 출력 예
SELECT word, type, detail FROM mecabko_analyze('인공지능모델학습데이터');  -- 복합명사

-- [2] 같은 문장을 PG 전문검색 설정별로 토큰화해 비교
--     simple : 공백·기호로만 자름 (조사가 붙은 채로 남는다)
--     korean : 형태소 분석 → 조사·어미 제거 → 원형
SELECT 'simple' AS cfg, to_tsvector('simple', '고양이가 사료를 먹었습니다. 고양이 사료 추천') AS tsv
UNION ALL
SELECT 'korean',        to_tsvector('korean', '고양이가 사료를 먹었습니다. 고양이 사료 추천');
-- 관찰: simple 에서는 '고양이가'와 '고양이'가 다른 토큰 → 키워드 검색이 서로 못 찾는다

-- [3] ts_debug: 토큰마다 어떤 사전이 처리했는지 (한글 → korean_stem, 영문 → english_stem)
SELECT alias, token, dictionaries, lexemes
FROM ts_debug('korean', 'PostgreSQL 17에서 벡터 검색을 테스트했다');

-- [4] tsvector 의 구조: 형태소 → 위치(position) 목록, 위치에 붙는 가중치(A/B/C/D)
SELECT * FROM unnest(to_tsvector('korean', '고양이 고양이 고양이 사료'));

-- [5] setweight: 구역별 가중치 딱지 붙이기 (강의 슬라이드 "PG ts_rank" 의 UPDATE 예제)
--     docs.tsv 는 생성 컬럼으로 제목=A, 본문=D 가 이미 적용돼 있다 (db/init/02_schema.sql)
SELECT setweight(to_tsvector('korean', '고양이 사료'), 'A')
    || setweight(to_tsvector('korean', '강아지 간식과 고양이 장난감'), 'D') AS weighted;

SELECT id, title, left(tsv::text, 200) AS tsv_head FROM docs ORDER BY id LIMIT 3;

-- [6] tsquery: 질의도 같은 형태소 분석을 거친다
SELECT plainto_tsquery('korean', '고양이 사료를 추천해 주세요');   -- 모든 단어 AND
SELECT websearch_to_tsquery('korean', '고양이 or 강아지 -간식');   -- 검색엔진 문법
SELECT to_tsquery('korean', '인공지능 <-> 모델');                 -- 인접(구문) 검색

-- [7] @@ 매칭 + GIN 인덱스 사용 확인
EXPLAIN ANALYZE
SELECT id, title FROM docs WHERE tsv @@ plainto_tsquery('korean', '반도체 수출');
