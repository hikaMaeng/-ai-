-- 실습용 본체 테이블
--   docs      : 검색 대상 지문(코퍼스). KLUE-MRC 의 context 를 중복 제거해 1행 1지문으로 저장
--   questions : 지문에 딸린 질문과 정답. "이 질문의 정답 지문이 상위 k 안에 오는가"로 검색 품질을 잰다

-- 지문의 형태소 분석 결과. 강의의 setweight 예제 그대로: 제목은 A, 본문은 D
--   적재할 때 CTE 에서 이 함수를 한 번 부르고, 그 결과로 tsv 와 doclen 을 함께 채운다
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
