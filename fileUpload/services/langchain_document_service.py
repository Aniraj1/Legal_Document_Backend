"""
LangChain-powered document processing service for PDF chunking and Upstash vector storage.
Replaces the old DocumentProcessor, ChunkingService, and VectorService.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import BytesIO

from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import UpstashVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class FixedDimensionEmbeddings(Embeddings):
    """Adapter to guarantee vectors match configured index dimension."""

    def __init__(self, base_embeddings: Embeddings, target_dimension: int):
        self.base_embeddings = base_embeddings
        self.target_dimension = int(target_dimension)

    def _fit_dimension(self, vector: List[float]) -> List[float]:
        if len(vector) == self.target_dimension:
            return vector
        if len(vector) > self.target_dimension:
            return vector[: self.target_dimension]
        return vector + [0.0] * (self.target_dimension - len(vector))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.base_embeddings.embed_documents(texts)
        return [self._fit_dimension(v) for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = self.base_embeddings.embed_query(text)
        return self._fit_dimension(vector)


class LangChainDocumentService:
    """
    Unified document loading, chunking, and vector storage using LangChain.
    
    Handles:
    - Document loading (PDF, DOCX, TXT)
    - Intelligent text splitting with semantic awareness
    - Real embeddings (Groq/OpenAI/HuggingFace)
    - Upstash vector store integration
    - Search and retrieval with metadata filtering
    """

    def __init__(self):
        """Initialize LangChain document service with embeddings and vector store"""
        try:
            self.chunk_size = settings.LANGCHAIN_CONFIG['CHUNK_SIZE']
            self.chunk_overlap = settings.LANGCHAIN_CONFIG['CHUNK_OVERLAP']
            self.embedding_model_name = settings.LANGCHAIN_CONFIG['EMBEDDING_MODEL']
            self.embedding_dimension = settings.LANGCHAIN_CONFIG['EMBEDDING_DIMENSION']
            
            self.embeddings = self._initialize_embeddings()
            self.vector_store = self._initialize_vector_store()
            
            logger.info(f"LangChainDocumentService initialized with {self.embedding_model_name} embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize LangChainDocumentService: {str(e)}")
            raise

    def _initialize_embeddings(self):
        """Initialize embeddings model based on configuration"""
        try:
            base_embeddings = None

            # Default to HuggingFace embeddings (free, local, no API needed)
            if self.embedding_model_name == 'groq':
                logger.warning("Groq doesn't support embeddings API. Using HuggingFace instead.")
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            elif self.embedding_model_name == 'openai':
                from importlib import import_module
                openai_module = import_module("langchain_openai")
                base_embeddings = openai_module.OpenAIEmbeddings(model="text-embedding-3-small")
            
            elif self.embedding_model_name == 'huggingface':
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            else:
                # Fallback to HuggingFace
                logger.warning(f"Unknown embedding model: {self.embedding_model_name}. Using HuggingFace.")
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            return FixedDimensionEmbeddings(base_embeddings, self.embedding_dimension)
        
        except ImportError as e:
            logger.error(f"Failed to import embeddings library: {str(e)}")
            raise ValueError(
                f"Please install the required package for {self.embedding_model_name} embeddings"
            )

    def _initialize_vector_store(self):
        """Initialize Upstash vector store"""
        try:
            url = settings.UPSTASH_VECTOR_REST_URL
            token = settings.UPSTASH_VECTOR_REST_TOKEN
            
            if not url or not token:
                raise ValueError("Missing Upstash configuration (URL or TOKEN)")

            return UpstashVectorStore(
                index_url=url,
                index_token=token,
                embedding=self.embeddings,
            )
        
        except ImportError:
            raise ImportError("upstash-vector package not installed. Run: pip install upstash-vector")
        except Exception as e:
            logger.error(f"Failed to initialize Upstash vector store: {str(e)}")
            raise

    @staticmethod
    def _save_temp_file(file_obj) -> str:
        """
        Save uploaded file to temporary location
        
        Args:
            file_obj: Django UploadedFile object
            
        Returns:
            str: Path to temporary file
        """
        import os
        
        try:
            file_ext = Path(file_obj.name).suffix
            temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
            
            file_obj.seek(0)
            os.write(temp_fd, file_obj.read())
            file_obj.seek(0)
            os.close(temp_fd)
            
            return temp_path
        
        except Exception as e:
            logger.error(f"Failed to save temp file: {str(e)}")
            raise

    def load_document(self, file_obj) -> List[Document]:
        """
        Load document and extract text with metadata.
        
        Supports: PDF, DOCX, DOC, TXT, and other formats via LangChain loaders
        
        Args:
            file_obj: Django UploadedFile object
            
        Returns:
            List[Document]: LangChain Document objects with content and metadata
            
        Raises:
            ValueError: If file type is not supported
            Exception: If document loading fails
        """
        file_ext = Path(file_obj.name).suffix.lower()
        
        try:
            # Save to temp file for LangChain loaders
            temp_path = self._save_temp_file(file_obj)
            
            try:
                if file_ext == '.pdf':
                    loader = PyPDFLoader(str(temp_path))
                
                elif file_ext == '.txt':
                    loader = TextLoader(str(temp_path))
                
                elif file_ext in ['.docx', '.doc']:
                    try:
                        from langchain_community.document_loaders import Docx2txtLoader
                        loader = Docx2txtLoader(str(temp_path))
                    except ImportError:
                        logger.warning("python-docx not available, treating as text file")
                        loader = TextLoader(str(temp_path))
                
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
                
                # Load documents
                documents = loader.load()
                
                if not documents:
                    raise ValueError("No content extracted from file")
                
                logger.info(f"Loaded {len(documents)} documents from {file_obj.name}")
                return documents
            
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
        
        except Exception as e:
            logger.error(f"Document loading failed for {file_obj.name}: {str(e)}")
            raise

    def chunk_document(
        self,
        documents: List[Document],
        user_id: str,
        file_id: str,
        file_name: str,
    ) -> List[Document]:
        """
        Chunk documents using RecursiveCharacterTextSplitter.
        
        Provides intelligent chunking that:
        - Respects content boundaries (paragraphs, sentences)
        - Uses semantic chunking hierarchy
        - Maintains metadata for tracking origin
        - Preserves context with overlap
        
        Args:
            documents: List of LangChain Document objects
            user_id: User ID for metadata
            file_id: File ID for metadata
            file_name: Original file name
            
        Returns:
            List[Document]: Chunked documents with enriched metadata
        """
        try:
            # Initialize recursive text splitter with semantic hierarchy
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",      # Paragraph boundaries
                    "\n",        # Line boundaries
                    ".",         # Sentence boundaries
                    " ",         # Word boundaries
                    "",          # Character (fallback)
                ],
                strip_whitespace=True,
            )
            
            # Split documents
            chunks = splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("No chunks created from documents")
            
            # Enrich chunks with metadata
            for chunk_index, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'user_id': str(user_id),
                    'file_id': str(file_id),
                    'file_name': file_name,
                    'chunk_index': chunk_index,
                    'total_chunks': len(chunks),
                    'word_count': len(chunk.page_content.split()),
                    'char_count': len(chunk.page_content),
                })
            
            logger.info(
                f"Created {len(chunks)} chunks from {len(documents)} documents "
                f"for file {file_name}"
            )
            return chunks
        
        except Exception as e:
            logger.error(f"Document chunking failed: {str(e)}")
            raise

    def upload_to_vector_store(
        self,
        chunks: List[Document],
        user_id: str,
        file_id: str,
    ) -> int:
        """
        Upload chunks to Upstash Vector Store with automatic batching.
        
        Features:
        - Automatic batching (default: 100 chunks per batch)
        - Efficient vector indexing
        - Error recovery
        - Progress tracking
        
        Args:
            chunks: List of chunked Document objects
            user_id: User ID for filtering
            file_id: File ID for filtering
            
        Returns:
            int: Number of successfully uploaded chunks
            
        Raises:
            Exception: If upload fails
        """
        try:
            if not chunks:
                logger.warning("No chunks to upload")
                return 0
            
            # Generate unique IDs for each chunk
            ids = [f"{user_id}:{file_id}:{i}" for i in range(len(chunks))]
            
            # LangChain handles batching automatically
            uploaded_ids = self.vector_store.add_documents(chunks, ids=ids)
            
            if not uploaded_ids:
                raise Exception("Vector store returned no IDs after upload")
            
            logger.info(
                f"Successfully uploaded {len(uploaded_ids)} chunks to Upstash "
                f"for file {file_id}"
            )
            return len(uploaded_ids)
        
        except Exception as e:
            logger.error(f"Vector upload failed: {str(e)}")
            raise

    def search(
        self,
        query: str,
        user_id: str,
        file_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Search vector store for relevant chunks with optional filtering.
        
        Performs semantic similarity search using embeddings.
        
        Args:
            query: Search query
            user_id: User ID for filtering
            file_id: Optional file ID for filtering to specific document
            top_k: Number of results to return (default: 5)
            
        Returns:
            List[tuple]: List of (Document, score) tuples, ordered by relevance
            
        Raises:
            Exception: If search fails
        """
        try:
            # UpstashVectorStore expects a filter expression string.
            def _quote(value: str) -> str:
                return str(value).replace("'", "\\'")

            filters = [f"user_id = '{_quote(user_id)}'"]
            if file_id:
                filters.append(f"file_id = '{_quote(file_id)}'")
            filter_expr = " AND ".join(filters)
            
            # Search with similarity scores
            results = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_expr,
            )
            
            if results:
                logger.info(
                    f"Found {len(results)} results for query in user {user_id} "
                    f"(file: {file_id or 'all'})"
                )
            else:
                logger.warning(f"No results found for query: {query}")
            
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise

    def delete_file_chunks(self, user_id: str, file_id: str) -> bool:
        """
        Delete all chunks for a specific file.
        
        Args:
            user_id: User ID
            file_id: File ID
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            Exception: If deletion fails
        """
        try:
            def _quote(value: str) -> str:
                return str(value).replace("'", "\\'")

            filter_expr = (
                f"user_id = '{_quote(user_id)}' AND file_id = '{_quote(file_id)}'"
            )

            # Upstash supports deletion by metadata filter.
            result = self.vector_store._index.delete(filter=filter_expr, namespace="")
            deleted = getattr(result, "deleted", 0)

            logger.info(
                f"Deleted {deleted} chunks for file {file_id} from user {user_id}"
            )
            return True
        
        except Exception as e:
            logger.error(f"Chunk deletion failed: {str(e)}")
            raise

    def delete_all_user_chunks(self, user_id: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Delete chunks for all provided files belonging to a user."""
        try:
            deleted_total = 0
            target_file_ids = [str(fid) for fid in (file_ids or [])]

            if not target_file_ids:
                return {
                    "success": True,
                    "deleted_count": 0,
                    "error": None,
                }

            for file_id in target_file_ids:
                self.delete_file_chunks(user_id=str(user_id), file_id=file_id)
                deleted_total += 1

            return {
                "success": True,
                "deleted_count": deleted_total,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Bulk chunk deletion failed for user {user_id}: {str(e)}")
            return {
                "success": False,
                "deleted_count": 0,
                "error": str(e),
            }

    def extract_text_with_metadata(self, file_obj) -> Dict[str, Any]:
        """
        Extract text from file with metadata (for backward compatibility).
        
        Args:
            file_obj: Django UploadedFile object
            
        Returns:
            Dict: Contains 'text' and 'metadata' keys
        """
        try:
            documents = self.load_document(file_obj)
            
            # Combine all document text
            full_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Extract metadata from first document
            first_metadata = documents[0].metadata if documents else {}
            
            return {
                'text': full_text,
                'metadata': {
                    'extraction_method': 'langchain_pdf_loader',
                    'pages': first_metadata.get('page', 1),
                    'word_count': len(full_text.split()),
                    'document_count': len(documents),
                }
            }
        
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise
