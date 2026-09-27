-- 컨테이너 최초 기동 시 1회 실행 (/docker-entrypoint-initdb.d)
CREATE EXTENSION IF NOT EXISTS vector;          -- pgvector
CREATE EXTENSION IF NOT EXISTS textsearch_ko;   -- mecab-ko 기반 한국어 전문검색 (parser/config: korean)

-- 이 DB 의 기본 전문검색 설정을 한국어로
DO $$ BEGIN
  EXECUTE format('ALTER DATABASE %I SET default_text_search_config = %L', current_database(), 'korean');
END $$;
