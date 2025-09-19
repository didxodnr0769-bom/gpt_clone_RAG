# gpt_clone_RAG

RAG(Retrieval-Augmented Generation) 기반의 챗봇 시스템으로, 로컬 LLM과 벡터 데이터베이스를 활용하여 문서 기반 질의응답을 제공합니다.

## 1. 프로젝트 설명

### 주요 기능

- **문서 업로드 및 인덱싱**: PDF, TXT, MD 등 다양한 형식의 문서를 업로드하고 벡터 데이터베이스에 저장
- **실시간 스트리밍 채팅**: Ollama 기반 로컬 LLM을 활용한 실시간 응답 생성
- **지능형 문서 청킹**: 문서 유형에 따른 자동 청킹 방법 선택 (semantic, markdown, recursive)
- **벡터 유사도 검색**: FAISS를 활용한 고성능 유사도 검색
- **중복 문서 관리**: 파일 해시 기반 중복 문서 감지 및 관리

### 기술 스택

- **백엔드**: FastAPI, Python
- **프론트엔드**: React, Vite, Tailwind CSS
- **LLM**: Ollama (qwen-ko-Q2, nomic-embed-text)
- **벡터 DB**: FAISS
- **문서 처리**: LangChain, Unstructured

## 2. 프로젝트 구조

```
gpt_clone_RAG/
├── client/                    # React 프론트엔드
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatInterface.jsx  # 채팅 인터페이스 컴포넌트
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── server/                    # FastAPI 백엔드
│   ├── api/
│   │   └── endpoints.py       # API 엔드포인트 정의
│   ├── core/
│   │   ├── rag_pipeline.py    # RAG 시스템 핵심 로직
│   │   └── file_loader.py     # 문서 로딩 및 청킹
│   ├── models/
│   │   └── schemas.py         # Pydantic 스키마 정의
│   ├── documents/             # 기본 문서 저장소
│   ├── uploads/               # 업로드된 파일 임시 저장소
│   ├── vector_store/          # FAISS 벡터 스토어
│   ├── main.py               # FastAPI 애플리케이션 진입점
│   ├── requirements.txt      # Python 의존성
│   └── start_server.sh       # 서버 시작 스크립트
└── README.md
```

### 주요 컴포넌트

- **RAGSystem**: 문서 인덱싱, 벡터 검색, LLM 응답 생성 담당
- **DocumentLoader**: 다양한 형식의 문서 로딩 및 청킹 처리
- **ChatInterface**: 실시간 스트리밍 채팅 UI
- **API Endpoints**: 문서 업로드, 채팅, 벡터 스토어 관리 API

## 3. 프로젝트 실행 방법

### 사전 요구사항

- Python 3.8+
- Node.js 16+
- Ollama 설치 및 실행

### 1. Ollama 설정

```bash
# Ollama 설치 (macOS)
brew install ollama

# Ollama 서비스 시작
ollama serve

# 필요한 모델 다운로드
ollama pull qwen-ko-Q2:latest
ollama pull nomic-embed-text:latest
```

### 2. 백엔드 서버 실행

```bash
# 서버 디렉토리로 이동
cd server

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 시작
./start_server.sh
# 또는
python main.py
```

서버는 `http://localhost:8000`에서 실행됩니다.

### 3. 프론트엔드 실행

```bash
# 클라이언트 디렉토리로 이동
cd client

# 의존성 설치
npm install

# 개발 서버 시작
npm run dev
```

프론트엔드는 `http://localhost:5173`에서 실행됩니다.

### 4. API 문서 확인

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 4. 프로젝트 참고 사항

### API 엔드포인트

- `POST /api/v1/chat`: 일반 채팅 (비스트리밍)
- `POST /api/v1/chat/stream`: 스트리밍 채팅
- `POST /api/v1/upload`: 문서 업로드
- `GET /api/v1/vector-store-info`: 벡터 스토어 정보 조회
- `GET /api/v1/chunking-methods`: 지원하는 청킹 방법 목록

### 문서 청킹 방법

1. **auto**: 문서 유형에 따라 자동 선택
2. **semantic**: 의미론적 유사성 기반 청킹
3. **markdown**: 마크다운 헤더 구조 기반 청킹
4. **recursive**: 재귀적 문자 분할

### 지원 파일 형식

- PDF (.pdf)
- 텍스트 (.txt)
- 마크다운 (.md)
- 워드 (.docx)
- 파워포인트 (.pptx)

### 성능 최적화

- FAISS 벡터 인덱싱으로 빠른 유사도 검색
- 문서 해시 기반 중복 제거
- 스트리밍 응답으로 사용자 경험 향상
- 로컬 LLM 사용으로 데이터 프라이버시 보장

### 주의사항

- Ollama 서비스가 실행 중이어야 함
- 벡터 스토어는 `server/vector_store/` 디렉토리에 저장됨
- 업로드된 파일은 `server/uploads/` 디렉토리에 임시 저장됨
- 대용량 문서 처리 시 메모리 사용량 주의
