import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router, set_rag_system
from core.rag_pipeline import RAGSystem

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 RAG 시스템 인스턴스
rag_system = None

app = FastAPI(
    title="RAG Chatbot API",
    description="RAG 기반 챗봇 백엔드 서버",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 이벤트"""
    global rag_system
    
    try:
        logger.info("RAG 시스템 초기화 중...")
        rag_system = RAGSystem()
        set_rag_system(rag_system)  # endpoints에 RAG 시스템 설정
        logger.info("RAG 시스템 초기화 완료")
        
        # documents 폴더 확인 및 생성
        documents_folder = "./documents"
        if not os.path.exists(documents_folder):
            os.makedirs(documents_folder)
            logger.info(f"documents 폴더 생성: {documents_folder}")
        else:
            logger.info(f"documents 폴더 확인: {documents_folder}")
        
        # documents 폴더 내 파일 목록 가져오기
        try:
            files = os.listdir(documents_folder)
            if not files:
                logger.info("documents 폴더가 비어있습니다.")
                return
            
            logger.info(f"문서 인덱싱 시작: {len(files)}개 파일 발견")
            
            # 각 파일에 대해 문서 인덱싱
            for file_name in files:
                file_path = os.path.join(documents_folder, file_name)
                
                # 파일인지 확인 (디렉토리 제외)
                if os.path.isfile(file_path):
                    logger.info(f"문서 인덱싱 중: {file_name}...")
                    
                    try:
                        result = rag_system.add_document(file_path)
                        if result["status"] == "success":
                            logger.info(f"✅ {file_name} 인덱싱 완료: {result['message']}")
                        else:
                            logger.error(f"❌ {file_name} 인덱싱 실패: {result['message']}")
                    except Exception as e:
                        logger.error(f"❌ {file_name} 인덱싱 중 오류 발생: {str(e)}")
                else:
                    logger.info(f"디렉토리 건너뛰기: {file_name}")
            
            # 벡터 스토어 정보 출력
            store_info = rag_system.get_vector_store_info()
            if store_info["status"] == "loaded":
                logger.info(f"📚 벡터 스토어 구축 완료: {store_info['document_count']}개 문서 인덱싱됨")
            else:
                logger.warning("⚠️ 벡터 스토어가 비어있습니다.")
                
        except Exception as e:
            logger.error(f"문서 인덱싱 중 오류 발생: {str(e)}")
            
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API 서버가 실행 중입니다."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# RAG 시스템 인스턴스를 endpoints에서 사용할 수 있도록 전역 변수로 제공
def get_rag_system():
    return rag_system

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
