import logging
import time
import re
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

    SYSTEM_PROMPT = """You are a helpful Legal assistant. Your role is to answer questions ONLY based on the provided document context.

CRITICAL RULES:
1. Only answer questions based on information in the provided context
2. If the answer is NOT in the context, explicitly state: "I could not find this information in the document"
3. Do NOT make up, infer, or assume information not in the context
4. Always cite which source/section your answer comes from
5. If context is unclear, say "The document doesn't provide clear information on this"
6. Be concise and factual
7. Keep your tone polite and helpful.

Document Context will follow. Answer ONLY from this context."""

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 700
    DEFAULT_TOP_K = 5
    DEFAULT_RETRIEVAL_CANDIDATES = 20
    MIN_RETRIEVAL_SCORE = 0.2
    MIN_CHUNKS_REQUIRED = 1
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'as',
        'and', 'or', 'but', 'if', 'then', 'than', 'that', 'this', 'these',
        'those', 'it', 'its', 'into', 'about', 'within', 'over', 'under',
        'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'can',
        'could', 'should', 'would', 'do', 'does', 'did', 'has', 'have', 'had',
        'please', 'show', 'tell', 'explain', 'summarize', 'summary'
    }

    POLITE_NO_CONTEXT_ANSWER = (
        "Sorry, I could not find that information in this document yet. "
        "Please ask about a specific clause, section, date, or party, and I will do my best to help."
    )

    def __init__(self):
        """Initialize the service with vector DB access via LangChain"""
        self.document_service = LangChainDocumentService()
        self.graph_rag_config = self._load_graph_rag_config()
        configured_max_tokens = getattr(settings, 'GROQ_MAX_TOKENS', self.DEFAULT_MAX_TOKENS)
        try:
            configured_max_tokens = int(configured_max_tokens)
        except (TypeError, ValueError):
            configured_max_tokens = self.DEFAULT_MAX_TOKENS

        # Keep generation budget within a practical guardrail.
        self.groq_max_tokens = max(128, min(configured_max_tokens, 4096))

    def _load_graph_rag_config(self):
        """Load Graph RAG configuration from settings with safe defaults."""
        defaults = {
            'ENABLED': True,
            'RETRIEVAL_CANDIDATES': self.DEFAULT_RETRIEVAL_CANDIDATES,
            'FINAL_TOP_K': self.DEFAULT_TOP_K,
            'MIN_HYBRID_SCORE': 0.35,
            'BASE_RANK_WEIGHT': 0.7,
            'TERM_MATCH_WEIGHT': 0.2,
            'SECTION_MATCH_WEIGHT': 0.1,
            'KEEP_FALLBACK_WHEN_EMPTY': True,
        }
        configured = getattr(settings, 'GRAPH_RAG_CONFIG', {}) or {}
        if not isinstance(configured, dict):
            return defaults
        merged = defaults.copy()
        merged.update(configured)
        return merged

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
            meta_answer = self._handle_meta_or_identity_query(query=query, file_obj=file_obj)
            if meta_answer:
                return {
                    'answer': meta_answer,
                    'sources': [],
                    'confidence': 'high',
                    'metadata': {
                        'file_id': str(file_id),
                        'file_name': file_obj.file_name,
                        'chunks_retrieved': 0,
                        'processing_time_ms': int((time.time() - start_time) * 1000),
                        'model_used': model,
                        'response_type': 'meta_or_identity'
                    },
                    'error': None
                }

            candidate_k = int(self.graph_rag_config.get('RETRIEVAL_CANDIDATES', self.DEFAULT_RETRIEVAL_CANDIDATES))
            chunks = self._retrieve_chunks(user_id, file_id, query, candidate_k)

            hybrid_enabled = bool(self.graph_rag_config.get('ENABLED', True))
            if hybrid_enabled:
                filtered_chunks, filter_summary = self._filter_and_rerank_chunks(chunks, query)
            else:
                filtered_chunks = chunks[:self.DEFAULT_TOP_K]
                filter_summary = {
                    'enabled': False,
                    'strategy': 'vector_only',
                    'raw_retrieved_count': len(chunks),
                    'kept_count': len(filtered_chunks),
                    'dropped_count': max(len(chunks) - len(filtered_chunks), 0),
                    'top_reasons': [],
                }

            self._log_retrieval_baseline(query=query, raw_chunks=chunks, filtered_chunks=filtered_chunks, summary=filter_summary)

            chat_history = chat_history or []
            # Step 3: Validate retrieval quality
            is_valid, quality_msg = self._validate_retrieval_quality(filtered_chunks)
            
            if not is_valid:
                return {
                    'answer': self.POLITE_NO_CONTEXT_ANSWER,
                    'sources': [],
                    'confidence': 'none',
                    'metadata': {
                        'file_id': str(file_id),
                        'file_name': file_obj.file_name,
                        'chunks_retrieved': 0,
                        'pre_filter_chunks': len(chunks),
                        'post_filter_chunks': len(filtered_chunks),
                        'filter_summary': filter_summary,
                        'processing_time_ms': int((time.time() - start_time) * 1000),
                        'model_used': model,
                        'validation_message': quality_msg
                    },
                    'error': None
                }
            
            # Step 4: Build context from chunks
            context = self._build_context(filtered_chunks)
            
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
                        'chunks_retrieved': len(filtered_chunks),
                        'pre_filter_chunks': len(chunks),
                        'post_filter_chunks': len(filtered_chunks),
                        'filter_summary': filter_summary,
                        'processing_time_ms': int((time.time() - start_time) * 1000),
                        'model_used': model
                    },
                    'error': 'Groq API returned empty response'
                }
            
            # Step 6: Extract citations from chunks
            sources = self._extract_citations(filtered_chunks)
            
            # Step 7: Determine confidence level
            confidence = self._calculate_confidence(filtered_chunks, answer)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'metadata': {
                    'file_id': str(file_id),
                    'file_name': file_obj.file_name,
                    'chunks_retrieved': len(filtered_chunks),
                    'pre_filter_chunks': len(chunks),
                    'post_filter_chunks': len(filtered_chunks),
                    'filter_summary': filter_summary,
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

    def _handle_meta_or_identity_query(self, query, file_obj):
        """
        Handle non-content questions that should not depend on retrieval quality.
        """
        normalized_query = (query or "").strip().lower()
        if not normalized_query:
            return None

        if re.search(r"\b(document\s*name|name\s*of\s*(the\s*)?document|file\s*name)\b", normalized_query):
            return f"The document name is '{file_obj.file_name}'."

        if re.search(r"\b(who\s*are\s*you|what\s*are\s*you|your\s*role)\b", normalized_query):
            return (
                "I am your document assistant. I can answer questions based on the uploaded file's content."
            )

        if re.search(r"\b(hello|hi|hey|good\s*(morning|afternoon|evening))\b", normalized_query):
            return "Hello. I am ready to help you with questions about this document."

        return None

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

    def _extract_query_terms(self, query):
        """Extract lightweight intent terms for graph-style reranking."""
        tokens = re.findall(r"[a-zA-Z0-9_]+", (query or '').lower())
        terms = []
        for token in tokens:
            if len(token) < 3:
                continue
            if token in self.STOPWORDS:
                continue
            terms.append(token)
        # Preserve order while removing duplicates.
        deduped = list(dict.fromkeys(terms))
        return deduped[:20]

    def _compute_chunk_hybrid_score(self, chunk, rank_index, total_count, query_terms):
        """Score chunk using vector rank + metadata/query overlap signals."""
        cfg = self.graph_rag_config
        metadata = chunk.get('metadata', {}) or {}
        text = (chunk.get('text') or '').lower()
        section_title = str(metadata.get('section_title', '')).lower()

        # Use retrieval order as robust base signal (works even if score semantics vary).
        denominator = max(total_count - 1, 1)
        rank_score = 1.0 - (rank_index / denominator)

        text_match_count = sum(1 for term in query_terms if term in text)
        section_match_count = sum(1 for term in query_terms if term and term in section_title)
        keyword_match_count = 0
        entity_match_count = 0

        keywords = metadata.get('keywords', [])
        entities = metadata.get('entities', [])
        if isinstance(keywords, list):
            keyword_blob = ' '.join(str(v).lower() for v in keywords)
            keyword_match_count = sum(1 for term in query_terms if term in keyword_blob)
        if isinstance(entities, list):
            entity_blob = ' '.join(str(v).lower() for v in entities)
            entity_match_count = sum(1 for term in query_terms if term in entity_blob)

        base_rank_weight = float(cfg.get('BASE_RANK_WEIGHT', 0.7))
        term_weight = float(cfg.get('TERM_MATCH_WEIGHT', 0.2))
        section_weight = float(cfg.get('SECTION_MATCH_WEIGHT', 0.1))

        term_bonus = min(text_match_count * 0.05, term_weight)
        section_bonus = min(section_match_count * 0.05, section_weight)
        metadata_bonus = min((keyword_match_count + entity_match_count) * 0.03, 0.15)
        short_text_penalty = 0.05 if len(text) < 80 else 0.0

        hybrid_score = (rank_score * base_rank_weight) + term_bonus + section_bonus + metadata_bonus - short_text_penalty
        reasons = []
        if text_match_count:
            reasons.append(f'text_matches:{text_match_count}')
        if section_match_count:
            reasons.append(f'section_matches:{section_match_count}')
        if keyword_match_count:
            reasons.append(f'keyword_matches:{keyword_match_count}')
        if entity_match_count:
            reasons.append(f'entity_matches:{entity_match_count}')
        if short_text_penalty > 0:
            reasons.append('short_text_penalty')
        if not reasons:
            reasons.append('rank_only')

        return hybrid_score, reasons

    def _filter_and_rerank_chunks(self, chunks, query):
        """Apply lightweight graph-style filtering and reranking."""
        if not chunks:
            return [], {
                'enabled': True,
                'strategy': 'hybrid_graph_rag_light',
                'raw_retrieved_count': 0,
                'kept_count': 0,
                'dropped_count': 0,
                'top_reasons': [],
            }

        query_terms = self._extract_query_terms(query)
        min_hybrid = float(self.graph_rag_config.get('MIN_HYBRID_SCORE', 0.35))
        final_top_k = int(self.graph_rag_config.get('FINAL_TOP_K', self.DEFAULT_TOP_K))
        keep_fallback = bool(self.graph_rag_config.get('KEEP_FALLBACK_WHEN_EMPTY', True))

        scored = []
        dropped = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            hybrid_score, reasons = self._compute_chunk_hybrid_score(
                chunk=chunk,
                rank_index=idx,
                total_count=total,
                query_terms=query_terms,
            )

            updated_chunk = dict(chunk)
            updated_chunk['hybrid_score'] = round(float(hybrid_score), 4)
            updated_chunk['selection_reasons'] = reasons

            if hybrid_score >= min_hybrid:
                scored.append(updated_chunk)
            else:
                dropped.append(updated_chunk)

        if not scored and keep_fallback:
            fallback = []
            for item in chunks[:final_top_k]:
                clone = dict(item)
                clone['hybrid_score'] = 0.0
                clone['selection_reasons'] = ['vector_fallback']
                fallback.append(clone)
            summary = {
                'enabled': True,
                'strategy': 'hybrid_graph_rag_light',
                'fallback_used': True,
                'raw_retrieved_count': len(chunks),
                'kept_count': len(fallback),
                'dropped_count': max(len(chunks) - len(fallback), 0),
                'query_terms': query_terms,
                'top_reasons': ['vector_fallback'],
            }
            return fallback, summary

        scored.sort(key=lambda c: c.get('hybrid_score', 0.0), reverse=True)
        kept = scored[:final_top_k]

        reason_counts = {}
        for chunk in kept:
            for reason in chunk.get('selection_reasons', []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        top_reasons = [
            reason
            for reason, _ in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        ]

        summary = {
            'enabled': True,
            'strategy': 'hybrid_graph_rag_light',
            'fallback_used': False,
            'raw_retrieved_count': len(chunks),
            'kept_count': len(kept),
            'dropped_count': len(dropped),
            'query_terms': query_terms,
            'top_reasons': top_reasons,
        }
        return kept, summary

    def _log_retrieval_baseline(self, query, raw_chunks, filtered_chunks, summary):
        """Emit baseline retrieval telemetry for phase-1 measurement."""
        logger.info(
            "[RetrievalBaseline] query_len=%s raw=%s filtered=%s strategy=%s fallback=%s",
            len(query or ''),
            len(raw_chunks or []),
            len(filtered_chunks or []),
            summary.get('strategy'),
            summary.get('fallback_used', False),
        )
        logger.info(
            "[RetrievalBaseline] top_reasons=%s query_terms=%s",
            summary.get('top_reasons', []),
            summary.get('query_terms', []),
        )

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
                "max_tokens": self.groq_max_tokens,
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
                'relevance_score': round(chunk.get('score', 0), 4),
                'hybrid_score': round(chunk.get('hybrid_score', 0), 4),
                'selection_reasons': chunk.get('selection_reasons', []),
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
        
        # Prefer hybrid score when present, otherwise fallback to raw score.
        top_score = chunks[0].get('hybrid_score', chunks[0].get('score', 0))
        
        if top_score >= 0.8:
            return 'high'
        elif top_score >= 0.6:
            return 'medium'
        else:
            return 'low'
