# ==============================================================================
# 시시콜콜 로컬 DB 환경 명령어
# ------------------------------------------------------------------------------
# 사전 준비:
#   1) Docker Desktop 설치
#   2) .env.example → .env 복사 후 값 확인
#   3) make up
# ==============================================================================

# .env 파일 로드 (POSTGRES_USER 등 변수 참조용)
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

.PHONY: help up down reset logs ps psql redis-cli chroma check seed seed-postgres seed-redis seed-chromadb

# ------------------------------------------------------------------------------
# 기본 명령
# ------------------------------------------------------------------------------

help:  ## 사용 가능한 명령어 목록
	@echo "시시콜콜 로컬 DB 환경 명령어"
	@echo ""
	@echo "  make up              # 3개 DB 컨테이너 기동 (백그라운드) + healthcheck"
	@echo "  make down            # 컨테이너 중지 (데이터는 유지됨)"
	@echo "  make reset           # 컨테이너 중지 + 볼륨 삭제 (완전 초기화)"
	@echo "  make logs            # 3개 서비스 로그 팔로우 (Ctrl+C로 종료)"
	@echo "  make ps              # 컨테이너 상태 확인"
	@echo "  make check           # 3개 DB healthcheck 일괄 검증"
	@echo ""
	@echo "  make psql            # PostgreSQL CLI 접속"
	@echo "  make redis-cli       # Redis CLI 접속"
	@echo "  make chroma          # ChromaDB heartbeat 확인"
	@echo ""
	@echo "  make seed            # 전체 시드 실행 (병원 + 음식점 tenant)"
	@echo "  make seed-postgres   # PostgreSQL 시드만"
	@echo "  make seed-redis      # Redis 시드만"
	@echo "  make seed-chromadb   # ChromaDB 컬렉션 시드만"

up:  ## 3개 DB 컨테이너 기동
	docker compose up -d
	@echo ""
	@echo "⏳ healthcheck 대기 중 (최대 30초)..."
	@sleep 8
	@$(MAKE) --no-print-directory check

down:  ## 컨테이너 중지 (데이터는 유지)
	docker compose down

reset:  ## 볼륨까지 완전 삭제
	docker compose down -v
	@echo ""
	@echo "⚠️  모든 데이터 볼륨이 삭제되었습니다."
	@echo "   다시 기동하려면: make up"

logs:  ## 3개 서비스 로그 팔로우
	docker compose logs -f

ps:  ## 컨테이너 상태 확인
	docker compose ps

# ------------------------------------------------------------------------------
# 개별 DB 접속
# ------------------------------------------------------------------------------

psql:  ## PostgreSQL CLI 접속
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

redis-cli:  ## Redis CLI 접속
	docker compose exec redis redis-cli

chroma:  ## ChromaDB heartbeat 확인
	@curl -sf http://localhost:$(CHROMA_PORT)/api/v1/heartbeat && echo "  ✅ ChromaDB alive" || echo "  ❌ ChromaDB 응답 없음"

# ------------------------------------------------------------------------------
# 통합 healthcheck
# ------------------------------------------------------------------------------

check:  ## 3개 DB healthcheck 일괄 검증
	@echo "▶ PostgreSQL:"
	@docker compose exec -T postgres pg_isready -U $(POSTGRES_USER) -d $(POSTGRES_DB) || (echo "  ❌ FAIL"; exit 1)
	@echo ""
	@echo "▶ Redis:"
	@docker compose exec -T redis redis-cli ping | grep -q PONG && echo "  ✅ PONG" || (echo "  ❌ FAIL"; exit 1)
	@echo ""
	@echo "▶ ChromaDB:"
	@curl -sf http://localhost:$(CHROMA_PORT)/api/v1/heartbeat > /dev/null && echo "  ✅ alive" || (echo "  ❌ FAIL"; exit 1)
	@echo ""
	@echo "🎉 3개 DB 모두 정상 동작 중"

# ------------------------------------------------------------------------------
# 시드 (Phase 3)
# ------------------------------------------------------------------------------

seed: seed-postgres seed-redis seed-chromadb  ## 전체 시드 실행
	@echo ""
	@echo "🌱 전체 시드 완료"

seed-postgres:
	python db/seed/seed_postgres.py

seed-redis:
	python db/seed/seed_redis.py

seed-chromadb:
	python db/seed/seed_chromadb.py
