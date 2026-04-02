from django.conf import settings
import hashlib


class VectorService:
    """Integrate with Upstash Vector Database"""

    VECTOR_DIMENSION = 1024  # Must match Upstash Vector index dimension

    def __init__(self):
        """Initialize Upstash Vector client"""
        try:
            from upstash_vector import Index
            
            self.index = Index(
                url=settings.UPSTASH_VECTOR_REST_URL,
                token=settings.UPSTASH_VECTOR_REST_TOKEN,
            )
        except ImportError:
            raise ImportError("upstash-vector package not installed. Run: pip install upstash-vector")
        except AttributeError as e:
            raise AttributeError(
                f"Missing Upstash configuration: {str(e)}. "
                "Set UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN in settings."
            )

    @staticmethod
    def _generate_simple_vector(text, dimension=1024):
        """
        Generate a deterministic vector from text by hashing
        Uses multiple hash iterations to create the required number of dimensions
        
        TODO: Replace with real embeddings (OpenAI, LLaMA, etc.) when available
        """
        import hashlib
        
        vector = []
        text_bytes = text.encode()
        
        # Generate vector by hashing text with different seeds
        for i in range(dimension):
            # Create hash with iteration counter as seed
            combined = text_bytes + str(i).encode()
            hash_obj = hashlib.sha256(combined)
            hash_hex = hash_obj.hexdigest()
            
            # Convert first 8 hex chars to float between 0-1
            hex_chunk = hash_hex[:8]
            value = int(hex_chunk, 16) / (2**32 - 1)
            vector.append(value)
        
        return vector

    def upload_chunks(self, chunks, user_id, file_id):
        """
        Upload chunks to Upstash Vector DB

        Args:
            chunks: list[dict] from ChunkingService with 'text' and 'metadata'
            user_id: User UUID
            file_id: File UUID (will use auto-generated if None)

        Returns:
            dict: {'success': bool, 'vector_ids': list, 'chunk_count': int, 'error': str}
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            vectors_to_upload = []

            for chunk in chunks:
                # Generate simple vector from chunk text
                vector = self._generate_simple_vector(chunk['text'], self.VECTOR_DIMENSION)
                
                # Prepare metadata with text included (so it returns in search results)
                metadata = chunk['metadata'].copy()
                metadata['text'] = chunk['text']  # Include actual text for retrieval
                
                # Create vector ID
                vector_id = f"{metadata['file_id']}_chunk_{metadata['chunk_index']}"
                
                # Upstash SDK format: (id, values, metadata)
                vector_data = (vector_id, vector, metadata)
                vectors_to_upload.append(vector_data)
                
                logger.debug(f"[Upload] Prepared chunk with ID={vector_id}, metadata={metadata}")

            # Batch upload to Upstash
            if vectors_to_upload:
                logger.info(f"[Upload] Uploading {len(vectors_to_upload)} vectors to Upstash")
                try:
                    result = self.index.upsert(vectors=vectors_to_upload)
                    logger.info(f"[Upload] Upstash upsert result: {result}")
                except TypeError as e:
                    # If tuple format fails, try dict format
                    logger.warning(f"[Upload] Tuple format failed: {e}, trying dict format")
                    vectors_as_dicts = []
                    for vector_id, vector, metadata in vectors_to_upload:
                        vectors_as_dicts.append({
                            "id": vector_id,
                            "values": vector,
                            "metadata": metadata
                        })
                    result = self.index.upsert(vectors=vectors_as_dicts)
                    logger.info(f"[Upload] Upstash upsert (dict format) result: {result}")

            vector_ids = [f"{chunks[i]['metadata']['file_id']}_chunk_{i}" for i in range(len(chunks))]

            logger.info(f"[Upload] Successfully uploaded {len(chunks)} chunks")
            return {
                'success': True,
                'vector_ids': vector_ids,
                'chunk_count': len(chunks),
                'error': None
            }

        except Exception as e:
            logger.error(f"[Upload] Error uploading chunks: {str(e)}", exc_info=True)
            return {
                'success': False,
                'vector_ids': [],
                'chunk_count': 0,
                'error': str(e)
            }

    def query(self, query_text, user_id=None, top_k=5):
        """
        Query documents in Upstash by vector similarity (basic)
        
        Note: Uses simple deterministic vectors, not semantic search
        For semantic search, implement with real embeddings (OpenAI, etc.)

        Args:
            query_text: Search text
            user_id: Optional user UUID to filter by user
            top_k: Number of results

        Returns:
            dict: {'success': bool, 'results': list, 'error': str}
        """
        try:
            # Generate vector for query
            query_vector = self._generate_simple_vector(query_text, self.VECTOR_DIMENSION)
            
            # Query Upstash (this SDK version does not support filter argument)
            # Always request metadata so callers can do Python-side filtering.
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )
            
            return {
                'success': True,
                'results': results,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'results': [],
                'error': str(e)
            }

    def search_document_chunks(self, user_id, field_id, query, top_k=5):
        """
        Search Upstash Vector DB for chunks relevant to a query, filtered by user and file.
        Used for RAG (Ask Groq) queries.
        
        Args:
            user_id: User UUID (string) - for access control
            field_id: File/Document UUID (string) - to search within specific document
            query: Search query text
            top_k: Number of results to return (default 5)
        
        Returns:
            list[dict]: List of chunks with text, metadata, and relevance scores
                [
                    {
                        'vector_id': str,
                        'text': str,
                        'metadata': dict,
                        'score': float
                    },
                    ...
                ]
        
        Raises:
            ValueError: If user_id or field_id is invalid
            ConnectionError: If Upstash is unavailable
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not user_id or not field_id:
            raise ValueError("user_id and field_id are required")
        
        try:
            # Generate vector for query using same method as chunks
            query_vector = self._generate_simple_vector(query, self.VECTOR_DIMENSION)
            
            user_id_str = str(user_id)
            field_id_str = str(field_id)
            
            # Query Upstash WITHOUT filter (filter param not supported in this SDK version)
            # We'll manually filter results in Python for security
            try:
                # Fetch even more to account for fallback extraction filtering
                fetch_k = max(top_k * 3, 15)  # Fetch 3x to increase chance of getting enough results
                logger.info(f"[VectorSearch] Querying Upstash with top_k={fetch_k} (requesting {top_k} results)")
                results = self.index.query(
                    vector=query_vector,
                    top_k=fetch_k,
                    include_metadata=True,
                )
                logger.debug(f"[VectorSearch] Query returned {type(results).__name__}: {results if not isinstance(results, list) else f'list of {len(results)}'}")
            except Exception as e:
                logger.error(f"[VectorSearch] Query failed: {str(e)}", exc_info=True)
                return []
            
            if not results:
                logger.warning(f"[VectorSearch] No results returned from Upstash for query")
                return []
            
            # Format and filter results
            formatted_results = []
            
            try:
                # Handle both list of results and single result
                if isinstance(results, list):
                    result_list = results
                    logger.debug(f"[VectorSearch] Results is a list with {len(result_list)} items")
                else:
                    # Single result or iterable
                    try:
                        result_list = list(results)
                        logger.debug(f"[VectorSearch] Results is iterable with {len(result_list)} items")
                    except TypeError:
                        # Single result object
                        result_list = [results]
                        logger.debug(f"[VectorSearch] Results is a single object")
                
                if not result_list:
                    logger.warning(f"[VectorSearch] Result list is empty")
                    return []
                
                logger.debug(f"[VectorSearch] Processing {len(result_list)} results from Upstash")
                
                for idx, result in enumerate(result_list):
                    try:
                        # Extract fields from QueryResult object
                        # QueryResult has attributes: id, data, metadata, score
                        vector_id = getattr(result, 'id', '')
                        metadata = getattr(result, 'metadata', None)
                        score = getattr(result, 'score', 0)

                        # Current upstash-vector SDK returns id/score/vector/metadata (no data field).
                        # Keep compatibility with prior payloads by deriving text from metadata.
                        text = ''
                        if metadata and isinstance(metadata, dict):
                            text = metadata.get('text', '')
                        
                        if text:
                            logger.debug(f"[VectorSearch] Result {idx}: Retrieved text from metadata ({len(text)} chars)")
                        
                        logger.debug(f"[VectorSearch] Result {idx}: id={vector_id}, text_len={len(text)}, has_metadata={bool(metadata)}, metadata_type={type(metadata).__name__}")
                        
                        # If metadata is None, try to extract from vector_id
                        if metadata is None or metadata == '':
                            logger.warning(f"[VectorSearch] Result {idx} has None metadata, attempting to extract from vector_id={vector_id}")
                            # Try to parse file_id from vector_id (format: file_id_chunk_index)
                            parts = vector_id.split('_chunk_')
                            if len(parts) == 2:
                                extracted_file_id = parts[0]
                                logger.debug(f"[VectorSearch] Extracted file_id from vector_id: {extracted_file_id}")
                                # Only include if file_id matches
                                if extracted_file_id == field_id_str:
                                    logger.debug(f"[VectorSearch] ✓ File ID match found - including result {idx}")
                                    formatted_result = {
                                        'vector_id': vector_id,
                                        'text': text,
                                        'metadata': {'file_id': extracted_file_id},
                                        'score': score
                                    }
                                    formatted_results.append(formatted_result)
                            continue
                        
                        # Handle metadata that might be JSON string
                        if isinstance(metadata, str):
                            try:
                                import json
                                metadata = json.loads(metadata)
                                logger.debug(f"[VectorSearch] Parsed JSON metadata")
                            except:
                                logger.warning(f"[VectorSearch] Metadata is string but not valid JSON: {metadata[:100]}")
                                metadata = {}
                        
                        # Manually filter by user_id and file_id for security
                        if metadata and isinstance(metadata, dict):
                            result_user_id = str(metadata.get('user_id', ''))
                            result_file_id = str(metadata.get('file_id', ''))
                            
                            logger.debug(f"[VectorSearch] Metadata keys: {list(metadata.keys())}")
                            logger.debug(f"[VectorSearch] Metadata: user_id='{result_user_id}', file_id='{result_file_id}'")
                            logger.debug(f"[VectorSearch] Expected: user_id='{user_id_str}', file_id='{field_id_str}'")
                            
                            # Normalize UUIDs for comparison (remove hyphens for comparison)
                            result_user_normalized = result_user_id.replace('-', '')
                            expected_user_normalized = user_id_str.replace('-', '')
                            result_file_normalized = result_file_id.replace('-', '')
                            expected_file_normalized = field_id_str.replace('-', '')
                            
                            # Only include results that match both user_id and file_id
                            if ((result_user_normalized == expected_user_normalized or result_user_id == user_id_str) and
                                (result_file_normalized == expected_file_normalized or result_file_id == field_id_str)):
                                logger.debug(f"[VectorSearch] ✓ Match found - including result {idx}")
                                formatted_result = {
                                    'vector_id': vector_id,
                                    'text': text,
                                    'metadata': metadata,
                                    'score': score
                                }
                                formatted_results.append(formatted_result)
                            else:
                                logger.debug(f"[VectorSearch] ✗ No match - skipping result {idx}")
                        else:
                            logger.warning(f"[VectorSearch] Result {idx} has no metadata or metadata is not a dict: type={type(metadata).__name__}")
                    except Exception as e:
                        logger.warning(f"[VectorSearch] Error processing result {idx}: {str(e)}", exc_info=True)
                        continue
                
                logger.info(f"[VectorSearch] Found {len(formatted_results)} matching chunks for user={user_id_str}, file={field_id_str} (filtered from {len(result_list)} total)")
                
                # If we got no results after filtering, log what was in the unfiltered results
                if not formatted_results and result_list:
                    logger.warning(f"[VectorSearch] DEBUGGING: Got {len(result_list)} unfiltered results but 0 matched the filter")
                    logger.warning(f"[VectorSearch] DEBUGGING: Showing first 3 results metadata:")
                    for idx in range(min(3, len(result_list))):
                        result = result_list[idx]
                        metadata = getattr(result, 'metadata', {})
                        logger.warning(f"[VectorSearch]   Result {idx}: {metadata}")
                
                # Return only top_k results
                return formatted_results[:top_k]
            
            except Exception as e:
                logger.error(f"[VectorSearch] Error formatting results: {str(e)}", exc_info=True)
                return []
            
        except Exception as e:
            logger.error(f"[VectorSearch] Vector search failed: {str(e)}")
            raise ConnectionError(f"Vector search failed: {str(e)}")
