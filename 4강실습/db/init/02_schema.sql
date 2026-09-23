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
    embedding vector(384)                       -- paraphrase-multilingual-MiniLM-L12-v2
);

CREATE INDEX docs_tsv_gin ON docs USING gin (tsv);

CREATE TABLE questions (
    id            bigserial PRIMARY KEY,
    guid          text UNIQUE NOT NULL,
    doc_id        bigint NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    question      text NOT NULL,
    answer        text,
    question_type int,                          -- 1: 패러프레이즈 2: 다문장 추론 3: 답 없음
    is_impossible boolean,
    embedding     vector(384)
);

CREATE INDEX questions_doc_id ON questions(doc_id);
