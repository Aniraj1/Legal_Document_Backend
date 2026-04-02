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
        try:
            vectors_to_upload = []

            for chunk in chunks:
                # Generate simple vector from chunk text
                vector = self._generate_simple_vector(chunk['text'], self.VECTOR_DIMENSION)
                
                # Prepare metadata
                metadata = chunk['metadata'].copy()
                
                # Create vector (id, values, metadata)
                # Using chunk index as unique ID within file
                vector_id = f"{metadata['file_id']}_chunk_{metadata['chunk_index']}"
                
                vector_data = (
                    vector_id,
                    vector,  # Simple deterministic vector
                    metadata
                )
                vectors_to_upload.append(vector_data)

            # Batch upload to Upstash
            if vectors_to_upload:
                self.index.upsert(vectors=vectors_to_upload)

            vector_ids = [f"{chunks[i]['metadata']['file_id']}_chunk_{i}" for i in range(len(chunks))]

            return {
                'success': True,
                'vector_ids': vector_ids,
                'chunk_count': len(chunks),
                'error': None
            }

        except Exception as e:
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
            
            # Prepare filter
            filter_dict = {}
            if user_id:
                filter_dict['user_id'] = str(user_id)
            
            # Query Upstash
            if filter_dict:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=filter_dict
                )
            else:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k
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
