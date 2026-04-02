import re


class ChunkingService:
    """Intelligent document chunking for legal documents"""

    TARGET_CHUNK_SIZE = 400  # estimated tokens
    MIN_CHUNK_SIZE = 10  # minimum words in a chunk (very lenient)
    
    @staticmethod
    def estimate_tokens(text):
        """
        Estimate token count (rough approximation)
        ~1.3 tokens per word on average for English
        """
        word_count = len(text.split())
        return int(word_count / 0.75)  # 1 token ≈ 0.75 words
    
    @staticmethod
    def should_skip_chunk(text):
        """Determine if chunk should be skipped (only skip truly empty or boilerplate)"""
        text_lower = text.lower().strip()
        
        # Skip only if empty or whitespace-only
        if not text_lower or len(text_lower.split()) < ChunkingService.MIN_CHUNK_SIZE:
            return True
        
        # Skip obvious boilerplate (very strict)
        boilerplate_patterns = [
            r'^©\s+\d{4}',  # Copyright only
            r'^all rights reserved\s*$',  # Exact match
            r'^—+\s*$',  # Just dashes
        ]
        
        if any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in boilerplate_patterns):
            return True
        
        return False

    @staticmethod
    def chunk_document(text, user_id=None, file_id=None, file_name=""):
        """
        Intelligently chunk document into meaningful sections

        Args:
            text: Full document text
            user_id: User UUID (for metadata)
            file_id: File UUID (for metadata - will use placeholder if None)
            file_name: Original file name

        Returns:
            list[dict]: Chunks with metadata
        """
        # Use placeholder IDs if not provided
        if file_id is None:
            file_id = "generated_at_save"
        if user_id is None:
            user_id = "unknown"

        if not text or not text.strip():
            return []
        
        # Step 1: Try to parse sections by headers
        sections = ChunkingService._parse_sections(text)
        
        # Step 2: Create chunks from sections
        chunks = []
        chunk_index = 0
        
        for section_title, section_text in sections:
            if not section_text or not section_text.strip():
                continue
            
            # If section is small enough, use as-is
            token_count = ChunkingService.estimate_tokens(section_text)
            
            if token_count <= ChunkingService.TARGET_CHUNK_SIZE:
                if not ChunkingService.should_skip_chunk(section_text):
                    chunk = {
                        'text': section_text.strip(),
                        'metadata': {
                            'user_id': str(user_id),
                            'file_id': str(file_id),
                            'chunk_index': chunk_index,
                            'section_title': section_title,
                            'token_count': token_count,
                            'word_count': len(section_text.split()),
                        }
                    }
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                # Split large sections by paragraphs
                sub_chunks = ChunkingService._split_section(
                    section_text, 
                    section_title, 
                    user_id, 
                    file_id, 
                    chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
        
        # Fallback: if no chunks created, create one from entire text
        if not chunks and text.strip():
            chunk = {
                'text': text.strip(),
                'metadata': {
                    'user_id': str(user_id),
                    'file_id': str(file_id),
                    'chunk_index': 0,
                    'section_title': 'Full Document',
                    'token_count': ChunkingService.estimate_tokens(text),
                    'word_count': len(text.split()),
                }
            }
            chunks.append(chunk)
        
        return chunks

    @staticmethod
    def _split_section(section_text, section_title, user_id, file_id, start_index):
        """Split a large section into smaller chunks"""
        chunks = []
        
        # Split by paragraphs (double newline)
        paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # If no paragraphs, split by single lines
            paragraphs = [line.strip() for line in section_text.split('\n') if line.strip()]
        
        if not paragraphs:
            return []
        
        current_chunk_text = ""
        chunk_index = start_index
        
        for para in paragraphs:
            current_plus_para = current_chunk_text + "\n\n" + para if current_chunk_text else para
            tokens_needed = ChunkingService.estimate_tokens(current_plus_para)
            
            if tokens_needed > ChunkingService.TARGET_CHUNK_SIZE:
                # Save current chunk if not empty
                if current_chunk_text.strip() and not ChunkingService.should_skip_chunk(current_chunk_text):
                    chunk = {
                        'text': current_chunk_text.strip(),
                        'metadata': {
                            'user_id': str(user_id),
                            'file_id': str(file_id),
                            'chunk_index': chunk_index,
                            'section_title': section_title,
                            'token_count': ChunkingService.estimate_tokens(current_chunk_text),
                            'word_count': len(current_chunk_text.split()),
                        }
                    }
                    chunks.append(chunk)
                    chunk_index += 1
                # Start new chunk
                current_chunk_text = para
            else:
                current_chunk_text = current_plus_para
        
        # Save last chunk
        if current_chunk_text.strip() and not ChunkingService.should_skip_chunk(current_chunk_text):
            chunk = {
                'text': current_chunk_text.strip(),
                'metadata': {
                    'user_id': str(user_id),
                    'file_id': str(file_id),
                    'chunk_index': chunk_index,
                    'section_title': section_title,
                    'token_count': ChunkingService.estimate_tokens(current_chunk_text),
                    'word_count': len(current_chunk_text.split()),
                }
            }
            chunks.append(chunk)
        
        return chunks

    @staticmethod
    def _parse_sections(text):
        """
        Parse document into sections by detecting headers
        Returns list of (section_title, section_text) tuples
        """
        sections = []
        lines = text.split('\n')
        
        current_section = "Untitled"
        current_text = ""
        
        for line in lines:
            # Detect section headers (# symbols, numbered, or all caps)
            header_match = re.match(r'^#+\s+(.+)$', line) or \
                          re.match(r'^([0-9]+\.?\s+[A-Z][A-Za-z\s]+)$', line) or \
                          re.match(r'^(§\s*[\d\.]+.+)$', line)
            
            if header_match:
                # Save previous section if it has content
                if current_text.strip():
                    sections.append((current_section, current_text.strip()))
                
                current_section = header_match.group(1).strip()
                current_text = ""
            else:
                current_text += line + "\n"
        
        # Save last section
        if current_text.strip():
            sections.append((current_section, current_text.strip()))
        
        return sections if sections else [("Untitled", text)]
