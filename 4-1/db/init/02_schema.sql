-- 실습용 본체 테이블
--   docs      : 검색 대상 지문(코퍼스). KLUE-MRC 의 context 를 중복 제거해 1행 1지문으로 저장
--   questions : 지문에 딸린 질문과 정답. "이 질문의 정답 지문이 상위 k 안에 오는가"로 검색 품질을 잰다

-- ── 사전 테이블 : 불용어 · 동의어 · 복합어를 한 테이블에서 kind 로 나눠 관리 ─────────────
--   한국어 불용어 · 동의어 목록은 PostgreSQL 에도 형태소 분석기(textsearch_ko)에도 없다.
--   분석기 파일(사전 파일 · mecab 사용자 사전)을 고치는 대신 SQL 테이블로 두고, 형태소 분석 앞뒤에서 적용한다.
--     compound : 원문에서 term 을 찾아 무조건 한 토큰으로 (분석 전 · 문맥에 따라 잘리는 것을 막음)
--     stop     : 분석 결과에서 term 형태소를 뺀다
--     syn      : 분석 결과의 term 형태소를 target 으로 바꾼다
--   편집은 INSERT · DELETE. 문서 쪽에 반영하려면 docs.tsv 를 다시 계산해야 한다(질의 쪽은 즉시)
CREATE TABLE lexicon (
    kind   text NOT NULL CHECK (kind IN ('stop', 'syn', 'compound')),
    term   text NOT NULL,
    target text,                        -- syn : 대표어(필수) · compound : 색인할 이름(없으면 term)
    PRIMARY KEY (kind, term),
    CHECK (kind <> 'syn'  OR target IS NOT NULL),
    CHECK (kind <> 'stop' OR target IS NULL)
);

-- 복합어 자리표시 토큰 : 형태소 분석기가 자르지 않는 영문 자음만의 한 단어 (모음이 없어 영어 어간 처리도 안 받는다)
CREATE FUNCTION lex_token(term text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE
RETURN 'qq' || translate(left(md5(term), 12), '0123456789abcdef', 'bcdfghjkmnpvwxzr') || 'q';

-- 분석 전 : 원문의 복합어를 자리표시 토큰으로 (긴 것부터)
CREATE FUNCTION lex_prep(t text) RETURNS text
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $$
DECLARE r record;
BEGIN
  FOR r IN SELECT term FROM lexicon WHERE kind = 'compound' ORDER BY length(term) DESC, term LOOP
    t := replace(t, r.term, ' ' || lex_token(r.term) || ' ');
  END LOOP;
  RETURN t;
END $$;

-- 분석 후 : 자리표시 → 복합어, 불용어 빼기, 동의어 → 대표어 (위치는 그대로)
--   바꿀 형태소가 하나도 없으면(사전이 비어 있을 때 포함) 입력을 그대로 돌려준다 — 다시 조립하는 비용을 아낌
CREATE FUNCTION lex_norm(v tsvector) RETURNS tsvector
LANGUAGE sql STABLE PARALLEL SAFE
RETURN CASE
  WHEN NOT EXISTS (SELECT 1 FROM unnest(tsvector_to_array(v)) l
                   JOIN lexicon x ON l = CASE WHEN x.kind = 'compound' THEN lex_token(x.term) ELSE x.term END)
    THEN v
  ELSE (
    SELECT coalesce(string_agg('''' || replace(replace(x.lx, '\', '\\'), '''', '''''') || ''':' || x.pos, ' '), '')::tsvector
    FROM (SELECT coalesce(s.target, b.lx) AS lx, b.pos
          FROM (SELECT coalesce(lower(coalesce(c.target, c.term)), t.lexeme) AS lx,
                       array_to_string(t.positions, ',') AS pos
                FROM unnest(v) t
                LEFT JOIN lexicon c ON c.kind = 'compound' AND lex_token(c.term) = t.lexeme) b
          LEFT JOIN lexicon s ON s.kind = 'syn' AND s.term = b.lx
          WHERE NOT EXISTS (SELECT 1 FROM lexicon st WHERE st.kind = 'stop' AND st.term = b.lx)) x)
END;

-- 문장 → 형태소 (문서 · 질의 공용). 사전 테이블이 비어 있으면 to_tsvector('korean', t) 와 같다
CREATE FUNCTION lex_tsvector(t text) RETURNS tsvector
LANGUAGE sql STABLE PARALLEL SAFE
RETURN lex_norm(to_tsvector('korean', lex_prep(t)));

-- 질의문 → tsquery (op : '&' 모두 포함 · '|' 하나라도)
CREATE FUNCTION lex_tsquery(q text, op text DEFAULT '&') RETURNS tsquery
LANGUAGE sql STABLE PARALLEL SAFE
RETURN (SELECT coalesce(string_agg('''' || replace(replace(l, '\', '\\'), '''', '''''') || '''', ' ' || op || ' '), '')::tsquery
        FROM unnest(tsvector_to_array(lex_tsvector(q))) l);

-- 지문의 형태소 분석 결과. 강의의 setweight 예제 그대로: 제목은 A, 본문은 D
--   적재할 때 CTE 에서 이 함수를 한 번 부르고, 그 결과로 tsv 와 doclen 을 함께 채운다
--   (실습1~6 은 분석기 그대로. 사전 테이블 lexicon 은 실습7 에서 lex_tsvector 로 쓴다)
CREATE FUNCTION doc_tsv(title text, content text) RETURNS tsvector
LANGUAGE sql IMMUTABLE PARALLEL SAFE
RETURN setweight(to_tsvector('korean', coalesce(title, '')), 'A') ||
       setweight(to_tsvector('korean', content), 'D');

-- 문서 길이 |D| = 형태소 등장 횟수의 합 (조사·어미가 빠진 형태소 수)
CREATE FUNCTION tsv_len(tsvector) RETURNS int
LANGUAGE sql IMMUTABLE PARALLEL SAFE
RETURN (SELECT coalesce(sum(cardinality(t.positions)), 0)::int FROM unnest($1) t);

CREATE TABLE docs (
    id        bigserial PRIMARY KEY,
    doc_key   text UNIQUE NOT NULL,             -- 지문 내용의 md5 (중복 제거용)
    title     text,
    content   text NOT NULL,
    category  text,                             -- 뉴스 분야 (위키 지문은 NULL)
    source    text,                             -- wikipedia / acrofan / hankyung ...
    -- 형태소 분석 결과와 문서 길이 : 넣는 쪽이 doc_tsv() 를 한 번 계산해 둘 다 채운다
    --   (생성 컬럼으로 두면 doclen 이 tsv 를 참조할 수 없어 형태소 분석이 두 번 돈다)
    --   제목·본문을 고칠 때도 tsv · doclen 을 같이 다시 넣어야 BM25 트리거가 역색인을 고친다
    tsv       tsvector NOT NULL,
    doclen    int      NOT NULL,                -- BM25 의 |D| = tsv_len(tsv)
    -- 임베딩 : Qwen3-Embedding-8B (LM Studio) 원본 4096차원. 재정렬·차원 비교용이라 인덱스 없음
    --   pgvector 의 HNSW·IVFFlat 인덱스는 vector 2,000차원까지라 4096 에는 만들 수 없다
    embedding vector(4096),
    -- MRL : 원본 앞 128차원을 잘라 다시 길이 1로. 후보 추리기용 HNSW 인덱스는 이 컬럼에
    --   잘라내면 길이가 원본보다 짧아진다(원본 1 → 약 0.2). 내적(<#>)으로 재도 코사인과 같게 정규화
    emb128    vector(128) GENERATED ALWAYS AS (
                  l2_normalize(subvector(embedding, 1, 128))::vector(128)
              ) STORED
);

CREATE INDEX docs_tsv_gin ON docs USING gin (tsv);
-- BM25 검색이 매번 avg(doclen) 과 문서별 doclen 을 읽는다 → 이 인덱스만 읽고 끝내게(인덱스 전용 스캔)
--   질문 500개 BM25 검색 : 인덱스 없음 7.0s → 있음 5.8s
CREATE INDEX docs_id_doclen ON docs (id) INCLUDE (doclen);
-- docs_emb128_hnsw 는 임베딩을 다 채운 뒤 적재 노트북이 만든다 (채우는 동안 인덱스를 갱신하면 느려진다)

CREATE TABLE questions (
    id            bigserial PRIMARY KEY,
    guid          text UNIQUE NOT NULL,
    doc_id        bigint NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    question      text NOT NULL,
    answer        text,
    question_type int,                          -- 1: 패러프레이즈 2: 다문장 추론 3: 답 없음
    is_impossible boolean,
    -- 질의 쪽이라 instruction 을 붙여 계산한다 (문서 쪽 docs.embedding 은 붙이지 않음)
    embedding     vector(4096),
    emb128        vector(128) GENERATED ALWAYS AS (
                      l2_normalize(subvector(embedding, 1, 128))::vector(128)
                  ) STORED
);

CREATE INDEX questions_doc_id ON questions(doc_id);
