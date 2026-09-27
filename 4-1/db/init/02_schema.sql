-- 실습용 본체 테이블
--   docs      : 검색 대상 지문(코퍼스). KLUE-MRC 의 context 를 중복 제거해 1행 1지문으로 저장
--   questions : 지문에 딸린 질문과 정답. "이 질문의 정답 지문이 상위 k 안에 오는가"로 검색 품질을 잰다

CREATE TABLE docs (
    id        bigserial PRIMARY KEY,
    doc_key   text UNIQUE NOT NULL,             -- 지문 내용의 md5 (중복 제거용)
    title     text,
    content   text NOT NULL,
    category  text,                             -- 뉴스 분야 (위키 지문은 NULL)
    source    text,                             -- wikipedia / acrofan / hankyung ...
    -- 강의의 setweight 예제 그대로: 제목은 A, 본문은 D
    -- 생성 컬럼이라 INSERT/UPDATE 때 자동 계산된다
    tsv       tsvector GENERATED ALWAYS AS (
                  setweight(to_tsvector('korean', coalesce(title, '')), 'A') ||
                  setweight(to_tsvector('korean', content), 'D')
              ) STORED,
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
