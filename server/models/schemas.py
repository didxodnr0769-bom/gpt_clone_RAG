from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    """채팅 요청 스키마"""
    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="세션 ID")
    context: Optional[str] = Field(None, description="추가 컨텍스트")

class SourceDocument(BaseModel):
    """소스 문서 스키마"""
    content: str = Field(..., description="문서 내용")
    metadata: Dict[str, Any] = Field(..., description="문서 메타데이터")
    score: Optional[float] = Field(None, description="관련도 점수")

class ChatResponse(BaseModel):
    """채팅 응답 스키마"""
    response: str = Field(..., description="챗봇 응답")
    sources: List[SourceDocument] = Field(default=[], description="참조된 소스 문서들")
    status: str = Field(..., description="응답 상태")
    timestamp: datetime = Field(default_factory=datetime.now, description="응답 생성 시간")

class StreamChunk(BaseModel):
    """스트림 청크 스키마"""
    chunk: str = Field(..., description="응답 청크")
    is_final: bool = Field(default=False, description="마지막 청크 여부")
    sources: Optional[List[SourceDocument]] = Field(None, description="참조된 소스 문서들 (마지막 청크에만 포함)")
    status: str = Field(default="streaming", description="스트림 상태")
    timestamp: datetime = Field(default_factory=datetime.now, description="청크 생성 시간")

class DocumentUploadResponse(BaseModel):
    """문서 업로드 응답 스키마"""
    message: str = Field(..., description="응답 메시지")
    uploaded_files: List[Dict[str, Any]] = Field(..., description="업로드된 파일 정보")
    processed_count: int = Field(..., description="처리된 문서 수")
    status: str = Field(..., description="처리 상태")

class ErrorResponse(BaseModel):
    """에러 응답 스키마"""
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보")
    status_code: int = Field(..., description="HTTP 상태 코드")

class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str = Field(..., description="서비스 상태")
    message: str = Field(..., description="상태 메시지")
    timestamp: datetime = Field(default_factory=datetime.now, description="체크 시간")

class SupportedFormatsResponse(BaseModel):
    """지원 파일 형식 응답 스키마"""
    supported_formats: List[str] = Field(..., description="지원하는 파일 형식 목록")
    message: str = Field(..., description="응답 메시지")

class DocumentUploadRequest(BaseModel):
    """문서 업로드 요청 스키마"""
    chunking_method: str = Field(default="auto", description="청킹 방법 (auto, semantic, markdown, recursive)")
    chunk_size: int = Field(default=500, description="청크 크기", ge=100, le=2000)
    chunk_overlap: int = Field(default=100, description="청크 겹침 크기", ge=0, le=500)

class ChunkingMethodInfo(BaseModel):
    """청킹 방법 정보 스키마"""
    method: str = Field(..., description="청킹 방법 이름")
    description: str = Field(..., description="청킹 방법 설명")
    best_for: List[str] = Field(..., description="최적 사용 사례")
