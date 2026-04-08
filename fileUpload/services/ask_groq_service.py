import logging
import time
from django.conf import settings
from fileUpload.services.langchain_document_service import LangChainDocumentService
from fileUpload.model.fileresources import FileResource

logger = logging.getLogger(__name__)


class AskGroqService:
    """
    RAG (Retrieval-Augmented Generation) service for answering questions about documents
    using Upstash Vector DB for retrieval and Groq API for generation.
    
    This service implements anti-hallucination strategies by:
    1. Enforcing retrieval-first approach
    2. Validating retrieval quality
    3. Using system prompts that constrain LLM to context
    4. Including source citations in response
    """

    SYSTEM_PROMPT = """You are a helpful document assistant. Your role is to answer questions ONLY based on the provided document context.

CRITICAL RULES:
1. Only answer questions based on information in the provided context
2. If the answer is NOT in the context, explicitly state: "I could not find this information in the document"
3. Do NOT make up, infer, or assume information not in the context
4. Always cite which source/section your answer comes from
5. If context is unclear, say "The document doesn't provide clear information on this"
6. Be concise and factual

Document Context will follow. Answer ONLY from this context."""

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 700
    DEFAULT_TOP_K = 5
    MIN_RETRIEVAL_SCORE = 0.3
    MIN_CHUNKS_REQUIRED = 2

    def __init__(self):
        """Initialize the service with vector DB access via LangChain"""
        self.document_service = LangChainDocumentService()

    def process_query(self, user_id, file_id, query, model=None, chat_history=None):
        """
        Main orchestration method for RAG pipeline.
        
        Args:
            user_id: User UUID (for authorization)
            file_id: File/Document UUID
            query: User question
            model: Groq model to use (optional, defaults to llama-3.1-8b-instant)
        
        Returns:
            dict: {
                'answer': str,
                'sources': list[dict],
                'confidence': str ('high', 'medium', 'low', 'none'),
                'metadata': dict,
                'error': str or None
            }
        """
        start_time = time.time()
        model = model or self.DEFAULT_MODEL
        
        try:
            # Step 1: Validate user has access to file
            file_obj = self._validate_file_access(user_id, file_id)
            if not file_obj:
                return {
                    'answer': None,
                    'sources': [],
                    'confidence': 'none',
                    'metadata': {},
                    'error': 'File not found or access denied'
                }
            
            # Step 2: Retrieve relevant chunks from vector DB
            chunks = self._retrieve_chunks(user_id, file_id, query, self.DEFAULT_TOP_K)
            chat_history = chat_history or []
            # Step 3: Validate retrieval quality
            is_valid, quality_msg = self._validate_retrieval_quality(chunks)
            
            if not is_valid:
                return {
                    'answer': "I could not find sufficient information about this in the document. The document does not appear to contain relevant details on this topic.",
                    'sources': [],
                    'confidence': 'none',
                    'metadata': {
                        'file_id': str(file_id),
                        'file_name': file_obj.file_name,
                        'chunks_retrieved': 0,
                        'processing_time_ms': int((time.time() - start_time) * 1000),
                        'model_used': model,
                        'validation_message': quality_msg
                    },
                    'error': None
                }
            
            # Step 4: Build context from chunks
            context = self._build_context(chunks)
            
            # Step 5: Call Groq API
            groq_start = time.time()
            answer = self._call_groq_api(query, context, file_obj.file_name, model, chat_history)
            groq_time = time.time() - groq_start
            
            if not answer:
                return {
                    'answer': "I could not generate a response. Please try again.",
                    'sources': [],
                    'confidence': 'none',
                    'metadata': {
                        'file_id': str(file_id),
                        'file_name': file_obj.file_name,
                        'chunks_retrieved': len(chunks),
                        'processing_time_ms': int((time.time() - start_time) * 1000),
                        'model_used': model
                    },
                    'error': 'Groq API returned empty response'
                }
            
            # Step 6: Extract citations from chunks
            sources = self._extract_citations(chunks)
            
            # Step 7: Determine confidence level
            confidence = self._calculate_confidence(chunks, answer)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'metadata': {
                    'file_id': str(file_id),
                    'file_name': file_obj.file_name,
                    'chunks_retrieved': len(chunks),
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'groq_time_ms': int(groq_time * 1000),
                    'model_used': model
                },
                'error': None
            }
            
        except Exception as e:
            logger.error(f"AskGroqService error: {str(e)}", exc_info=True)
            return {
                'answer': None,
                'sources': [],
                'confidence': 'none',
                'metadata': {},
                'error': f'Service error: {str(e)}'
            }

    def _validate_file_access(self, user_id, file_id):
        """
        Verify user has access to the requested file.
        
        Returns:
            FileResource object if access granted, None otherwise
        """
        try:
            file_obj = FileResource.objects.get(id=file_id, user_id=user_id)
            return file_obj
        except FileResource.DoesNotExist:
            logger.warning(f"Unauthorized file access attempt: user={user_id}, file={file_id}")
            return None
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            return None

    def _retrieve_chunks(self, user_id, file_id, query, top_k=5):
        """
        Query Upstash Vector DB for relevant chunks using LangChain service.
        
        Returns:
            list[dict]: Top K chunks with metadata and scores
        """
        try:
            logger.debug(f"[AskGroqService] Retrieving chunks for user={user_id}, file={file_id}, query_length={len(query)}")
            
            # Use LangChain service to search
            results = self.document_service.search(
                query=query,
                user_id=str(user_id),
                file_id=str(file_id),
                top_k=top_k
            )
            
            # Convert LangChain Document objects with scores to dict format
            chunks = []
            for doc, score in results:
                chunks.append({
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score,
                })
            
            logger.info(f"[AskGroqService] Retrieved {len(chunks)} chunks")
            
            return chunks if chunks else []
        except Exception as e:
            logger.error(f"[AskGroqService] Vector retrieval error: {str(e)}", exc_info=True)
            return []

    def _validate_retrieval_quality(self, chunks, min_results=None):
        """
        Validate that retrieval found sufficient context.
        
        Returns:
            tuple: (is_valid: bool, message: str)
        """
        min_results = min_results or self.MIN_CHUNKS_REQUIRED
        
        if not chunks or len(chunks) < min_results:
            return False, f"Insufficient chunks retrieved: {len(chunks)}/{min_results}"
        
        # Check if top results have reasonable scores
        top_chunk = chunks[0]
        score = top_chunk.get('score', 0)
        
        if score < self.MIN_RETRIEVAL_SCORE:
            return False, f"Low relevance score: {score:.2f}"
        
        return True, "Retrieval quality validated"

    def _build_context(self, chunks):
        """
        Format retrieved chunks as context string with citations.
        
        Returns:
            str: Formatted context
        """
        context_parts = []
        
        logger.info(f"[AskGroqService] Building context from {len(chunks)} chunks")
        
        for idx, chunk in enumerate(chunks, 1):
            section_title = chunk.get('metadata', {}).get('section_title', f'Section {idx}')
            text = chunk.get('text', '').strip()
            score = chunk.get('score', 0)
            
            # Log the chunk being included
            logger.debug(f"[AskGroqService] Chunk {idx}: title='{section_title}', score={score:.4f}, text_length={len(text)}")
            
            # Format: [Source Label] <score indicator> \n content
            score_indicator = f"(Relevance: {score:.2%})" if score else ""
            context_parts.append(f"[{section_title}] {score_indicator}\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"[AskGroqService] Context built: {len(context)} characters total")
        
        return context

    def _call_groq_api(self, query, context, filename, model, chat_history=None):
        """
        Call Groq API with system prompts that enforce grounding.
        
        Returns:
            str: LLM response
        """
        try:
            import requests
            
            groq_key = settings.GROQ_API_KEY
            if not groq_key:
                raise ValueError("GROQ_API_KEY not configured in settings")
            
            # Log the query and context being sent to Groq
            logger.info(f"[AskGroqService] Preparing Groq API call")
            logger.debug(f"[AskGroqService] Query: '{query}'")
            logger.debug(f"[AskGroqService] Filename: '{filename}'")
            logger.debug(f"[AskGroqService] Context length: {len(context)} characters")
            logger.debug(f"[AskGroqService] Model: '{model}'")
            
            context_prompt = f"""Document: {filename}

RETRIEVED CONTEXT:
{context}

Remember: Answer ONLY from the above context. Do not add external knowledge."""
            
            # Include a bounded amount of prior turns for follow-up questions.
            recent_history = (chat_history or [])[-6:]
            history_messages = [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in recent_history
                if isinstance(message, dict)
                and message.get("role") in ["user", "assistant"]
                and isinstance(message.get("content"), str)
                and message.get("content").strip()
            ]

            messages = [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "system",
                    "content": context_prompt
                },
            ]
            messages.extend(history_messages)
            messages.append({
                "role": "user",
                "content": query
            })

            payload = {
                "model": model,
                "temperature": self.DEFAULT_TEMPERATURE,
                "max_tokens": self.DEFAULT_MAX_TOKENS,
                "top_p": 0.9,
                "messages": messages
            }
            
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"[AskGroqService] Sending request to Groq API ({self.GROQ_API_URL})")
            start_time = time.time()
            
            response = requests.post(self.GROQ_API_URL, json=payload, headers=headers, timeout=30)
            elapsed = time.time() - start_time
            
            logger.info(f"[AskGroqService] Groq API response received in {elapsed:.2f}s (status: {response.status_code})")
            
            if not response.ok:
                error_msg = response.text
                logger.error(f"[AskGroqService] Groq API error {response.status_code}: {error_msg}")
                raise Exception(f"Groq API error {response.status_code}")
            
            data = response.json()
            answer = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            logger.debug(f"[AskGroqService] Generated answer length: {len(answer)} characters")
            logger.debug(f"[AskGroqService] Generated answer preview: {answer[:200]}..." if len(answer) > 200 else answer)
            
            return answer if answer else None
            
        except Exception as e:
            logger.error(f"[AskGroqService] Groq API call failed: {str(e)}")
            return None

    def _extract_citations(self, chunks):
        """
        Extract source citations from retrieved chunks.
        
        Returns:
            list[dict]: Sources with metadata
        """
        sources = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            # Show up to 150 chars of actual content
            content_preview = chunk.get('text', '')[:150]
            if len(chunk.get('text', '')) > 150:
                content_preview += "..."
            
            source = {
                'chunk_id': chunk.get('vector_id', ''),
                'section_title': metadata.get('section_title', 'Unknown'),
                'content_preview': content_preview,
                'relevance_score': round(chunk.get('score', 0), 4)
            }
            sources.append(source)
        
        return sources

    def _calculate_confidence(self, chunks, answer):
        """
        Calculate confidence level based on retrieval quality and answer content.
        
        Returns:
            str: 'high', 'medium', 'low', 'none'
        """
        if not chunks:
            return 'none'
        
        if not answer or answer.startswith("I could not find"):
            return 'low'
        
        # Check if top chunk has high relevance score
        top_score = chunks[0].get('score', 0)
        
        if top_score >= 0.8:
            return 'high'
        elif top_score >= 0.6:
            return 'medium'
        else:
            return 'low'
