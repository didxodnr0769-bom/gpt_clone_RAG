import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

class DocumentLoader:
    """문서 로딩 및 처리 로직을 담당하는 클래스"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx', '.md']
    
    async def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """단일 문서를 로드하고 처리"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
            
            # 파일 형식에 따른 로딩 로직
            if file_extension == '.txt':
                return await self._load_text_file(file_path)
            elif file_extension == '.pdf':
                return await self._load_pdf_file(file_path)
            elif file_extension == '.docx':
                return await self._load_docx_file(file_path)
            elif file_extension == '.md':
                return await self._load_markdown_file(file_path)
            
        except Exception as e:
            logger.error(f"문서 로딩 실패: {file_path}, 오류: {e}")
            raise
    
    async def _load_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """텍스트 파일 로딩"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return [{
                "content": content,
                "metadata": {
                    "file_path": file_path,
                    "file_type": "text",
                    "file_size": os.path.getsize(file_path)
                }
            }]
        except Exception as e:
            logger.error(f"텍스트 파일 로딩 실패: {e}")
            raise
    
    async def _load_pdf_file(self, file_path: str) -> List[Dict[str, Any]]:
        """PDF 파일 로딩"""
        # TODO: PyPDF2 또는 pdfplumber를 사용한 PDF 파싱 구현
        logger.info(f"PDF 파일 로딩: {file_path}")
        return [{
            "content": "PDF 내용이 여기에 표시됩니다.",
            "metadata": {
                "file_path": file_path,
                "file_type": "pdf",
                "file_size": os.path.getsize(file_path)
            }
        }]
    
    async def _load_docx_file(self, file_path: str) -> List[Dict[str, Any]]:
        """DOCX 파일 로딩"""
        # TODO: python-docx를 사용한 DOCX 파싱 구현
        logger.info(f"DOCX 파일 로딩: {file_path}")
        return [{
            "content": "DOCX 내용이 여기에 표시됩니다.",
            "metadata": {
                "file_path": file_path,
                "file_type": "docx",
                "file_size": os.path.getsize(file_path)
            }
        }]
    
    async def _load_markdown_file(self, file_path: str) -> List[Dict[str, Any]]:
        """마크다운 파일 로딩"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return [{
                "content": content,
                "metadata": {
                    "file_path": file_path,
                    "file_type": "markdown",
                    "file_size": os.path.getsize(file_path)
                }
            }]
        except Exception as e:
            logger.error(f"마크다운 파일 로딩 실패: {e}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """지원하는 파일 형식 목록 반환"""
        return self.supported_formats

def load_and_chunk_document_semantic(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    의미론적 청킹을 사용하여 문서를 로드하고 텍스트 청크로 분할하는 함수
    
    Args:
        file_path (str): 로드할 파일 경로
        chunk_size (int): 청크 크기 (기본값: 500)
        chunk_overlap (int): 청크 간 겹치는 부분 크기 (기본값: 100)
    
    Returns:
        List[Dict[str, Any]]: 텍스트 청크들의 리스트
    """
    try:
        logger.info(f"의미론적 문서 로딩 및 청킹 시작: {file_path}")
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # UnstructuredFileLoader를 사용하여 문서 로드
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        
        if not documents:
            logger.warning(f"문서에서 내용을 추출할 수 없습니다: {file_path}")
            return []
        
        # 임베딩 모델 초기화 (의미론적 청킹용)
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        
        # 의미론적 청킹을 위한 SemanticChunker 사용
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        
        # 문서를 의미론적으로 청크로 분할
        chunks = semantic_splitter.split_documents(documents)
        
        # 청크를 딕셔너리 형태로 변환
        chunk_list = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "content": chunk.page_content,
                "metadata": {
                    "source": file_path,
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content),
                    "chunking_method": "semantic",
                    **chunk.metadata
                }
            }
            chunk_list.append(chunk_dict)
        
        logger.info(f"의미론적 문서 청킹 완료: {file_path}, 총 {len(chunk_list)}개 청크 생성")
        return chunk_list
        
    except Exception as e:
        logger.error(f"의미론적 문서 로딩 및 청킹 실패: {file_path}, 오류: {e}")
        raise

def load_and_chunk_document_markdown(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    마크다운 문서의 헤더 구조를 고려한 청킹 함수
    
    Args:
        file_path (str): 로드할 파일 경로
        chunk_size (int): 청크 크기 (기본값: 500)
        chunk_overlap (int): 청크 간 겹치는 부분 크기 (기본값: 100)
    
    Returns:
        List[Dict[str, Any]]: 텍스트 청크들의 리스트
    """
    try:
        logger.info(f"마크다운 구조 기반 문서 로딩 및 청킹 시작: {file_path}")
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # UnstructuredFileLoader를 사용하여 문서 로드
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        
        if not documents:
            logger.warning(f"문서에서 내용을 추출할 수 없습니다: {file_path}")
            return []
        
        # 마크다운 헤더 기반 분할기 설정
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        # 일반 텍스트 분할기 (헤더 분할 후 추가 분할용)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 문서를 헤더 기반으로 먼저 분할
        md_header_splits = markdown_splitter.split_text(documents[0].page_content)
        
        # 각 헤더 섹션을 추가로 청크로 분할
        all_chunks = []
        for split in md_header_splits:
            chunks = text_splitter.split_text(split.page_content)
            for chunk in chunks:
                all_chunks.append({
                    "page_content": chunk,
                    "metadata": {**split.metadata, **documents[0].metadata}
                })
        
        # 청크를 딕셔너리 형태로 변환
        chunk_list = []
        for i, chunk in enumerate(all_chunks):
            chunk_dict = {
                "content": chunk.page_content,
                "metadata": {
                    "source": file_path,
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content),
                    "chunking_method": "markdown_header",
                    **chunk.metadata
                }
            }
            chunk_list.append(chunk_dict)
        
        logger.info(f"마크다운 구조 기반 문서 청킹 완료: {file_path}, 총 {len(chunk_list)}개 청크 생성")
        return chunk_list
        
    except Exception as e:
        logger.error(f"마크다운 구조 기반 문서 로딩 및 청킹 실패: {file_path}, 오류: {e}")
        raise

def load_and_chunk_document(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    LangChain을 사용하여 문서를 로드하고 텍스트 청크로 분할하는 함수
    
    Args:
        file_path (str): 로드할 파일 경로
        chunk_size (int): 청크 크기 (기본값: 1000)
        chunk_overlap (int): 청크 간 겹치는 부분 크기 (기본값: 200)
    
    Returns:
        List[Dict[str, Any]]: 텍스트 청크들의 리스트
    """
    try:
        logger.info(f"문서 로딩 및 청킹 시작: {file_path}")
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # UnstructuredFileLoader를 사용하여 문서 로드
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        
        if not documents:
            logger.warning(f"문서에서 내용을 추출할 수 없습니다: {file_path}")
            return []
        
        # RecursiveCharacterTextSplitter를 사용하여 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 문서를 청크로 분할
        chunks = text_splitter.split_documents(documents)
        
        # 청크를 딕셔너리 형태로 변환
        chunk_list = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "content": chunk.page_content,
                "metadata": {
                    "source": file_path,
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content),
                    **chunk.metadata
                }
            }
            chunk_list.append(chunk_dict)
        
        logger.info(f"문서 청킹 완료: {file_path}, 총 {len(chunk_list)}개 청크 생성")
        return chunk_list
        
    except Exception as e:
        logger.error(f"문서 로딩 및 청킹 실패: {file_path}, 오류: {e}")
        raise

def load_and_chunk_document_adaptive(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100, method: str = "auto") -> List[Dict[str, Any]]:
    """
    문서 유형에 따라 적응적으로 청킹 방법을 선택하는 함수
    
    Args:
        file_path (str): 로드할 파일 경로
        chunk_size (int): 청크 크기 (기본값: 500)
        chunk_overlap (int): 청크 간 겹치는 부분 크기 (기본값: 100)
        method (str): 청킹 방법 ("auto", "semantic", "markdown", "recursive")
    
    Returns:
        List[Dict[str, Any]]: 텍스트 청크들의 리스트
    """
    try:
        file_extension = Path(file_path).suffix.lower()
        
        # 자동 선택 모드
        if method == "auto":
            if file_extension == '.md':
                logger.info(f"마크다운 파일 감지, 헤더 기반 청킹 사용: {file_path}")
                return load_and_chunk_document_markdown(file_path, chunk_size, chunk_overlap)
            else:
                logger.info(f"일반 파일 감지, 의미론적 청킹 사용: {file_path}")
                return load_and_chunk_document_semantic(file_path, chunk_size, chunk_overlap)
        
        # 수동 선택 모드
        elif method == "semantic":
            return load_and_chunk_document_semantic(file_path, chunk_size, chunk_overlap)
        elif method == "markdown":
            return load_and_chunk_document_markdown(file_path, chunk_size, chunk_overlap)
        elif method == "recursive":
            return load_and_chunk_document(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"지원하지 않는 청킹 방법입니다: {method}")
            
    except Exception as e:
        logger.error(f"적응적 문서 로딩 및 청킹 실패: {file_path}, 오류: {e}")
        raise
