from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from typing import List
import logging
import os
import json
from models.schemas import ChatRequest, ChatResponse, DocumentUploadResponse, StreamChunk, DocumentUploadRequest, ChunkingMethodInfo

logger = logging.getLogger(__name__)

# API 라우터 생성
router = APIRouter()

# RAG 시스템 인스턴스를 저장할 전역 변수
_rag_system = None

def set_rag_system(rag_system):
    """RAG 시스템 인스턴스를 설정하는 함수"""
    global _rag_system
    _rag_system = rag_system

def get_rag_system():
    """RAG 시스템 인스턴스를 반환하는 함수"""
    return _rag_system

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """챗봇과 대화하는 엔드포인트"""
    try:
        logger.info(f"채팅 요청 수신: {request.message}")
        
        # RAG 시스템 인스턴스 가져오기
        rag_system = get_rag_system()
        if rag_system is None:
            raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")
        
        # RAG 시스템을 통한 응답 생성
        result = rag_system.query(request.message)
        
        if result["status"] == "success":
            return ChatResponse(
                response=result["answer"],
                sources=result.get("sources", []),
                status="success"
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"채팅 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}")

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """스트림 방식으로 챗봇과 대화하는 엔드포인트"""
    try:
        logger.info(f"스트림 채팅 요청 수신: {request.message}")
        
        # RAG 시스템 인스턴스 가져오기
        rag_system = get_rag_system()
        if rag_system is None:
            raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")
        
        def generate_stream():
            """스트림 응답 생성기"""
            try:
                for chunk_data in rag_system.query_stream(request.message):
                    # StreamChunk 객체 생성
                    stream_chunk = StreamChunk(
                        chunk=chunk_data.get("chunk", ""),
                        is_final=chunk_data.get("is_final", False),
                        sources=chunk_data.get("sources"),
                        status=chunk_data.get("status", "streaming")
                    )
                    
                    # 프론트엔드 친화적인 JSON 스트림 형태로 전송
                    yield stream_chunk.model_dump_json() + "\n"
                    
                    # 마지막 청크인 경우 스트림 종료
                    if chunk_data.get("is_final", False):
                        break
                        
            except Exception as e:
                logger.error(f"스트림 생성 중 오류: {e}")
                error_chunk = StreamChunk(
                    chunk=f"스트림 처리 중 오류가 발생했습니다: {str(e)}",
                    is_final=True,
                    status="error"
                )
                yield error_chunk.model_dump_json() + "\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/json; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
        
    except Exception as e:
        logger.error(f"스트림 채팅 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"스트림 채팅 처리 중 오류가 발생했습니다: {str(e)}")

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunking_method: str = "auto",
    chunk_size: int = 10000,
    chunk_overlap: int = 100,
    replace_existing: bool = True
):
    """문서 업로드 엔드포인트"""
    try:
        # RAG 시스템 인스턴스 가져오기
        rag_system = get_rag_system()
        if rag_system is None:
            raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")
        
        uploaded_files = []
        processed_count = 0
        
        # uploads 폴더 생성
        uploads_folder = "uploads"
        os.makedirs(uploads_folder, exist_ok=True)
        
        for file in files:
            # 파일 저장
            file_path = os.path.join(uploads_folder, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append({
                "filename": file.filename,
                "file_path": file_path,
                "file_size": len(content)
            })
            
            # RAG 시스템에 문서 추가 (선택된 청킹 방법 사용)
            try:
                result = rag_system.add_document(
                    file_path, 
                    chunking_method=chunking_method,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    replace_existing=replace_existing
                )
                if result["status"] == "success":
                    processed_count += result.get("chunks_count", 0)
                    logger.info(f"문서 추가 완료: {file.filename} (청킹 방법: {chunking_method})")
                elif result["status"] == "skipped":
                    logger.info(f"문서 건너뛰기: {file.filename} - {result['message']}")
                else:
                    logger.error(f"문서 추가 실패: {file.filename} - {result['message']}")
            except Exception as e:
                logger.error(f"문서 처리 중 오류: {file.filename} - {str(e)}")
        
        return DocumentUploadResponse(
            message=f"문서 업로드 및 처리가 완료되었습니다. ({processed_count}개 청크 처리됨)",
            uploaded_files=uploaded_files,
            processed_count=processed_count,
            status="success"
        )
            
    except Exception as e:
        logger.error(f"문서 업로드 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"문서 업로드 중 오류가 발생했습니다: {str(e)}")

@router.get("/health")
async def health_check():
    """API 상태 확인 엔드포인트"""
    return {"status": "healthy", "message": "API가 정상적으로 작동 중입니다."}

@router.get("/supported-formats")
async def get_supported_formats():
    """지원하는 파일 형식 목록 반환"""
    from core.file_loader import DocumentLoader
    loader = DocumentLoader()
    return {
        "supported_formats": loader.get_supported_formats(),
        "message": "지원하는 파일 형식 목록입니다."
    }

@router.get("/vector-store-info")
async def get_vector_store_info():
    """벡터 스토어 정보 조회"""
    try:
        rag_system = get_rag_system()
        if rag_system is None:
            return {"status": "error", "message": "RAG 시스템이 초기화되지 않았습니다."}
        
        info = rag_system.get_vector_store_info()
        return info
    except Exception as e:
        logger.error(f"벡터 스토어 정보 조회 실패: {e}")
        return {"status": "error", "message": f"벡터 스토어 정보 조회 실패: {str(e)}"}

@router.get("/chunking-methods", response_model=List[ChunkingMethodInfo])
async def get_chunking_methods():
    """사용 가능한 청킹 방법 목록 반환"""
    return [
        ChunkingMethodInfo(
            method="auto",
            description="문서 유형에 따라 자동으로 최적의 청킹 방법을 선택합니다",
            best_for=["모든 문서 유형", "자동화된 처리"]
        ),
        ChunkingMethodInfo(
            method="semantic",
            description="의미론적 유사성을 기반으로 문서를 청크로 분할합니다",
            best_for=["긴 문서", "의미론적 일관성이 중요한 문서", "학술 논문", "기술 문서"]
        ),
        ChunkingMethodInfo(
            method="markdown",
            description="마크다운 헤더 구조를 고려하여 문서를 청크로 분할합니다",
            best_for=["마크다운 파일", "구조화된 문서", "README 파일", "기술 문서"]
        ),
        ChunkingMethodInfo(
            method="recursive",
            description="재귀적 문자 분할을 사용하여 문서를 청크로 분할합니다",
            best_for=["일반 텍스트", "간단한 문서", "빠른 처리"]
        )
    ]
