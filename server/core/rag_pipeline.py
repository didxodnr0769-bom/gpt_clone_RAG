from typing import List, Dict, Any, Generator
import logging
import os
import pickle
import hashlib
from pathlib import Path
from .file_loader import DocumentLoader, load_and_chunk_document, load_and_chunk_document_adaptive
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

logger = logging.getLogger(__name__)

class StreamCallbackHandler(BaseCallbackHandler):
    """스트림 콜백 핸들러"""
    
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """새로운 토큰이 생성될 때 호출"""
        self.tokens.append(token)
    
    def get_tokens(self) -> List[str]:
        """생성된 토큰들 반환"""
        return self.tokens.copy()
    
    def clear_tokens(self):
        """토큰 리스트 초기화"""
        self.tokens.clear()

class RAGSystem:
    """RAG 시스템을 관리하는 클래스"""
    
    def __init__(self, vector_store_path: str = "vector_store"):
        """
        RAG 시스템 초기화
        
        Args:
            vector_store_path (str): 벡터 스토어 저장 경로
        """
        self.vector_store_path = vector_store_path
        self.document_loader = DocumentLoader()
        
        # 문서 해시 추적을 위한 딕셔너리
        self.document_hashes = {}
        self._load_document_hashes()
        
        # 임베딩 모델 초기화 (Ollama nomic-embed-text 모델)
        logger.info("Ollama 임베딩 모델 로딩 중...")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest"
        )
        
        # Ollama LLM 초기화
        logger.info("Ollama LLM 초기화 중...")
        self.llm = Ollama(model="qwen-ko-Q2:latest")
        
        # FAISS 벡터 스토어 초기화
        self.vector_store = None
        self._load_or_create_vector_store()
        
        # 프롬프트 템플릿 설정
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""다음 문서들을 참고하여 질문에 답변해주세요.

문서 내용:
{context}

질문: {question}

답변:"""
        )
        
        logger.info("RAG 시스템 초기화 완료")
    
    def _load_document_hashes(self):
        """저장된 문서 해시 정보 로드"""
        hash_file = os.path.join(self.vector_store_path, "document_hashes.pkl")
        try:
            if os.path.exists(hash_file):
                with open(hash_file, 'rb') as f:
                    self.document_hashes = pickle.load(f)
                logger.info(f"문서 해시 정보 로드 완료: {len(self.document_hashes)}개 문서")
            else:
                self.document_hashes = {}
                logger.info("새로운 문서 해시 추적 시작")
        except Exception as e:
            logger.error(f"문서 해시 로드 실패: {e}")
            self.document_hashes = {}
    
    def _save_document_hashes(self):
        """문서 해시 정보 저장"""
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            hash_file = os.path.join(self.vector_store_path, "document_hashes.pkl")
            with open(hash_file, 'wb') as f:
                pickle.dump(self.document_hashes, f)
            logger.info("문서 해시 정보 저장 완료")
        except Exception as e:
            logger.error(f"문서 해시 저장 실패: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일의 해시값 계산"""
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            return hashlib.md5(file_content).hexdigest()
        except Exception as e:
            logger.error(f"파일 해시 계산 실패: {file_path}, 오류: {e}")
            return ""
    
    def _is_document_duplicate(self, file_path: str) -> bool:
        """문서가 이미 벡터 스토어에 있는지 확인"""
        file_hash = self._calculate_file_hash(file_path)
        return file_hash in self.document_hashes.values()
    
    def _remove_document_from_vector_store(self, file_path: str):
        """벡터 스토어에서 특정 문서 제거"""
        try:
            if self.vector_store is None:
                return
            
            # 해당 파일의 해시 찾기
            file_hash = self._calculate_file_hash(file_path)
            if file_hash not in self.document_hashes.values():
                return
            
            # 벡터 스토어에서 해당 파일의 문서들 찾아서 제거
            # FAISS는 직접적인 문서 제거를 지원하지 않으므로, 
            # 새로운 벡터 스토어를 생성하는 방식으로 처리
            logger.info(f"문서 제거 중: {file_path}")
            
            # 현재 벡터 스토어의 모든 문서를 가져와서 해당 파일 제외
            all_docs = self.vector_store.similarity_search("", k=10000)  # 충분히 큰 k 값
            filtered_docs = []
            
            for doc in all_docs:
                if doc.metadata.get("source") != file_path:
                    filtered_docs.append(doc)
            
            if filtered_docs:
                # 필터링된 문서들로 새 벡터 스토어 생성
                self.vector_store = FAISS.from_documents(filtered_docs, self.embeddings)
            else:
                # 모든 문서가 제거된 경우
                self.vector_store = None
            
            # 해시에서도 제거
            keys_to_remove = [k for k, v in self.document_hashes.items() if v == file_hash]
            for key in keys_to_remove:
                del self.document_hashes[key]
            
            logger.info(f"문서 제거 완료: {file_path}")
            
        except Exception as e:
            logger.error(f"문서 제거 실패: {file_path}, 오류: {e}")
    
    def _load_or_create_vector_store(self):
        """기존 벡터 스토어 로드 또는 새로 생성"""
        try:
            if os.path.exists(self.vector_store_path):
                logger.info("기존 벡터 스토어 로딩 중...")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("벡터 스토어 로딩 완료")
            else:
                logger.info("새로운 벡터 스토어 생성 중...")
                # 빈 벡터 스토어 생성
                self.vector_store = None
        except Exception as e:
            logger.error(f"벡터 스토어 로딩 실패: {e}")
            self.vector_store = None
    
    def _save_vector_store(self):
        """벡터 스토어를 디스크에 저장"""
        try:
            if self.vector_store is not None:
                os.makedirs(self.vector_store_path, exist_ok=True)
                self.vector_store.save_local(self.vector_store_path)
                # 문서 해시 정보도 함께 저장
                self._save_document_hashes()
                logger.info("벡터 스토어 저장 완료")
        except Exception as e:
            logger.error(f"벡터 스토어 저장 실패: {e}")
            raise
    
    def add_document(self, file_path: str, chunking_method: str = "recursive", chunk_size: int = 100, chunk_overlap: int = 10, replace_existing: bool = True) -> Dict[str, Any]:
        """
        문서를 벡터 DB에 추가하고 인덱싱
        
        Args:
            file_path (str): 추가할 문서 파일 경로
            chunking_method (str): 청킹 방법 ("auto", "semantic", "markdown", "recursive")
            chunk_size (int): 청크 크기
            chunk_overlap (int): 청크 겹침 크기
            replace_existing (bool): 기존 문서가 있을 때 교체할지 여부
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            logger.info(f"문서 추가 시작: {file_path}")
            
            # 중복 문서 체크
            if self._is_document_duplicate(file_path):
                if replace_existing:
                    logger.info(f"기존 문서 발견, 교체 중: {file_path}")
                    self._remove_document_from_vector_store(file_path)
                else:
                    logger.info(f"중복 문서 발견, 건너뛰기: {file_path}")
                    return {
                        "status": "skipped",
                        "message": "이미 존재하는 문서입니다. 교체하려면 replace_existing=True로 설정하세요.",
                        "chunks_count": 0
                    }
            
            # 문서 로딩 및 청킹 (선택된 방법 사용)
            chunks = load_and_chunk_document_adaptive(
                file_path, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                method=chunking_method
            )
            
            if not chunks:
                return {
                    "status": "error",
                    "message": "문서에서 내용을 추출할 수 없습니다."
                }
            
            # LangChain Document 객체로 변환
            from langchain.schema import Document
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"]
                )
                documents.append(doc)
            
            # 벡터 스토어에 추가
            if self.vector_store is None:
                # 첫 번째 문서인 경우 새 벡터 스토어 생성
                self.vector_store = FAISS.from_documents(
                    documents, 
                    self.embeddings
                )
            else:
                # 기존 벡터 스토어에 추가
                new_vector_store = FAISS.from_documents(
                    documents, 
                    self.embeddings
                )
                self.vector_store.merge_from(new_vector_store)
            
            # 문서 해시 추가
            file_hash = self._calculate_file_hash(file_path)
            self.document_hashes[file_path] = file_hash
            
            # 벡터 스토어 저장
            self._save_vector_store()
            
            logger.info(f"문서 추가 완료: {file_path}, {len(chunks)}개 청크")
            return {
                "status": "success",
                "message": f"문서가 성공적으로 추가되었습니다. ({len(chunks)}개 청크)",
                "chunks_count": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {file_path}, 오류: {e}")
            return {
                "status": "error",
                "message": f"문서 추가 중 오류가 발생했습니다: {str(e)}"
            }
    
    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        사용자 질문에 대한 답변 생성
        
        Args:
            question (str): 사용자 질문
            k (int): 검색할 관련 문서 수 (기본값: 4)
            
        Returns:
            Dict[str, Any]: 답변 및 관련 문서 정보
        """
        try:
            logger.info(f"질문 처리 시작: {question}")
            
            if self.vector_store is None:
                return {
                    "status": "error",
                    "message": "벡터 스토어가 비어있습니다. 먼저 문서를 추가해주세요."
                }
            
            # 관련 문서 검색
            relevant_docs = self.vector_store.similarity_search(
                question, 
                k=k
            )
            
            if not relevant_docs:
                return {
                    "status": "error",
                    "message": "관련 문서를 찾을 수 없습니다."
                }
            
            # 검색된 문서들을 컨텍스트로 결합
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 프롬프트 생성
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # LLM을 사용한 답변 생성
            response = self.llm(prompt)
            
            # 소스 문서 정보 생성
            sources = []
            for i, doc in enumerate(relevant_docs):
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": i + 1  # 간단한 관련도 점수
                })
            
            logger.info(f"질문 처리 완료: {question}")
            return {
                "status": "success",
                "answer": response,
                "sources": sources,
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {question}, 오류: {e}")
            return {
                "status": "error",
                "message": f"질문 처리 중 오류가 발생했습니다: {str(e)}"
            }
    
    def query_stream(self, question: str) -> Generator[Dict[str, Any], None, None]:
        """
        스트림 방식으로 질문 처리
        
        Args:
            question (str): 사용자 질문
            
        Yields:
            Dict[str, Any]: 스트림 청크 데이터
        """
        try:
            if self.vector_store is None:
                yield {
                    "chunk": "벡터 스토어가 비어있습니다. 먼저 문서를 추가해주세요.",
                    "is_final": True,
                    "sources": [],
                    "status": "error"
                }
                return
            
            logger.info(f"스트림 질문 처리 시작: {question}")
            
            # 관련 문서 검색 (검색 개수 증가)
            relevant_docs = self.vector_store.similarity_search(question, k=10)
            
            # 조 번호가 포함된 질문의 경우 키워드 검색도 추가
            import re
            article_pattern = r'제\s*\d+\s*조'
            if re.search(article_pattern, question):
                logger.info(f"조 번호 감지, 키워드 검색 추가: {question}")
                # 모든 문서에서 해당 조 번호 검색
                all_docs = self.vector_store.similarity_search("", k=10000)
                keyword_docs = []
                for doc in all_docs:
                    if re.search(article_pattern, doc.page_content):
                        # 질문의 조 번호와 매칭되는 문서 찾기
                        question_articles = re.findall(article_pattern, question)
                        doc_articles = re.findall(article_pattern, doc.page_content)
                        if any(article in doc_articles for article in question_articles):
                            keyword_docs.append(doc)
                
                # 키워드 검색 결과를 기존 결과와 결합
                if keyword_docs:
                    relevant_docs = keyword_docs + relevant_docs
                    # 중복 제거
                    seen = set()
                    unique_docs = []
                    for doc in relevant_docs:
                        doc_id = id(doc)
                        if doc_id not in seen:
                            seen.add(doc_id)
                            unique_docs.append(doc)
                    relevant_docs = unique_docs[:10]  # 상위 10개로 제한
            
            if not relevant_docs:
                yield {
                    "chunk": "관련 문서를 찾을 수 없습니다.",
                    "is_final": True,
                    "sources": [],
                    "status": "error"
                }
                return
            
            # 컨텍스트 생성
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 프롬프트 생성
            prompt = self.prompt_template.format(context=context, question=question)
            
            # 스트림 콜백 핸들러 생성
            stream_handler = StreamCallbackHandler()
            
            # LLM에 스트림 콜백 설정
            self.llm.callbacks = [stream_handler]
            
            # 스트림 응답 생성
            response = ""
            for chunk in self.llm.stream(prompt):
                if chunk:
                    response += chunk
                    yield {
                        "chunk": chunk,
                        "is_final": False,
                        "sources": None,
                        "status": "streaming"
                    }
            
            # 소스 문서 정보 생성
            sources = []
            for i, doc in enumerate(relevant_docs):
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": i + 1
                })
            
            # 최종 청크 (소스 정보 포함)
            yield {
                "chunk": "",
                "is_final": True,
                "sources": sources,
                "status": "success"
            }
            
            logger.info(f"스트림 질문 처리 완료: {question}")
            
        except Exception as e:
            logger.error(f"스트림 질문 처리 실패: {question}, 오류: {e}")
            yield {
                "chunk": f"질문 처리 중 오류가 발생했습니다: {str(e)}",
                "is_final": True,
                "sources": [],
                "status": "error"
            }
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """벡터 스토어 정보 반환"""
        if self.vector_store is None:
            return {
                "status": "empty",
                "message": "벡터 스토어가 비어있습니다.",
                "tracked_documents": list(self.document_hashes.keys())
            }
        
        try:
            # 벡터 스토어의 문서 수 확인
            doc_count = self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else 0
            
            return {
                "status": "loaded",
                "document_count": doc_count,
                "vector_store_path": self.vector_store_path,
                "tracked_documents": list(self.document_hashes.keys()),
                "tracked_count": len(self.document_hashes)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"벡터 스토어 정보 조회 실패: {str(e)}"
            }
