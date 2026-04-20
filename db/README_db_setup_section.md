<!-- ============================================================================ -->
<!-- 이 섹션을 기존 README.md 에 복사해서 "로컬 개발 환경 구성" 부근에 붙여넣으세요.  -->
<!-- ============================================================================ -->

## 🐳 로컬 DB 환경 구성

시시콜콜은 **PostgreSQL · Redis · ChromaDB** 세 개의 데이터 저장소를 사용합니다.  
팀원 간 환경 통일을 위해 **Docker Compose** 로 일괄 관리합니다.

### 사전 준비

| 항목 | 설명 |
|---|---|
| Docker Desktop | [공식 다운로드](https://www.docker.com/products/docker-desktop/) — macOS / Windows / Linux 모두 지원 |
| Make | macOS는 기본 포함 · Windows는 WSL2 또는 `choco install make` |

> **Windows 사용자**: Docker Desktop 설치 시 WSL2 백엔드 선택 권장. Windows Home Edition 도 WSL2 활성화 후 사용 가능합니다.

### 3단계 온보딩

```bash
# 1) .env 파일 생성 (최초 1회)
cp .env.example .env
#    → .env 열어서 필요한 값 수정 (DB 비밀번호 등)

# 2) DB 컨테이너 기동
make up
#    → PostgreSQL(5432) + Redis(6379) + ChromaDB(8001) 자동 기동
#    → 최초 기동 시 db/init/*.sql 자동 실행으로 9개 테이블 생성

# 3) 시드 데이터 주입 (병원 + 음식점 tenant)
make seed
```

`make up` 이 성공하면 마지막 줄에 `🎉 3개 DB 모두 정상 동작 중` 메시지가 표시됩니다.

### 주요 명령어

| 명령 | 용도 |
|---|---|
| `make up` | 3개 DB 기동 + healthcheck |
| `make down` | 컨테이너 중지 (데이터는 유지됨) |
| `make reset` | **볼륨까지 완전 삭제** — 스키마 변경 후 재적용 시 사용 |
| `make check` | 3개 DB 상태 일괄 확인 |
| `make logs` | 실시간 로그 팔로우 |
| `make psql` | PostgreSQL CLI 접속 |
| `make redis-cli` | Redis CLI 접속 |
| `make chroma` | ChromaDB heartbeat 확인 |
| `make seed` | 시드 데이터 주입 (멱등) |

### 포트 할당

| 서비스 | 호스트 포트 | 컨테이너 내부 | 비고 |
|---|---|---|---|
| PostgreSQL | 5432 | 5432 | |
| Redis | 6379 | 6379 | AOF persistence 활성화 |
| ChromaDB | **8001** | 8000 | FastAPI 8000 충돌 회피 |
| FastAPI (앱) | 8000 | - | Docker 외부에서 직접 실행 |

### 데이터 초기화 정책

| 상황 | 명령 | 데이터 |
|---|---|---|
| 스키마 변경 없음, 재시작만 필요 | `make down` → `make up` | 유지 |
| 스키마 변경 후 적용 필요 | `make reset` → `make up` → `make seed` | **전체 삭제 후 재생성** |

> **중요**: `db/init/*.sql` 은 **최초 기동 시에만** 실행됩니다.  
> 이미 데이터가 있는 상태에서 DDL을 수정해도 반영되지 않으므로, **스키마 변경 시 반드시 `make reset` 필요**합니다.

### 시드 데이터 범위

M1 단계의 최소 시드입니다. 연구 팀원들의 병목을 막는 목적입니다.

| 저장소 | 시드 내용 |
|---|---|
| PostgreSQL `tenants` | 서울중앙병원 (+821000000001) · 한밭식당 (+821000000002) |
| PostgreSQL `knn_intents` | 각 tenant 별 10~13개 예시 문장 (**더미 임베딩 1024d**) |
| Redis | 각 tenant 별 `business_hours` / `agent_availability` Hash |
| ChromaDB | 각 tenant 별 빈 컬렉션 (`tenant_{uuid}_docs`) |

> ⚠️ **KNN intent 임베딩은 더미 제로 벡터**입니다.  
> 희영(BGE-M3 연구) 완료 후 별도 재계산 스크립트로 교체 예정입니다.  
> 그 전까지 KNN Router 는 정상 동작하지 않습니다 (신용 연구 시 의도된 제약).

### 시드 의존성

`make seed` 실행 전 Python 패키지 설치 필요:

```bash
pip install -r db/seed/requirements.txt
```

메인 `requirements.txt` 에 이미 `asyncpg` / `redis` / `chromadb` / `python-dotenv` 가 포함되어 있다면 별도 설치는 불필요합니다.

### 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `make up` 후 healthcheck 실패 | 포트 충돌 — `lsof -i :5432,6379,8001` 확인 후 기존 프로세스 종료 |
| ChromaDB healthcheck timeout | `docker compose logs chromadb` 로 로그 확인. 디스크 여유공간 부족 의심 |
| `make seed` 시 `DATABASE_URL` 없음 에러 | `.env` 파일이 프로젝트 루트에 있는지 확인 |
| 스키마 변경 후 반영 안 됨 | `make reset` → `make up` → `make seed` 순서로 재적용 |
| Windows에서 `make` 명령 없음 | WSL2 터미널 사용 또는 `choco install make` 설치 |

---

<!-- ============================================================================ -->
<!-- 끝: 로컬 DB 환경 구성 섹션                                                   -->
<!-- ============================================================================ -->
