from pathlib import Path
from io import BytesIO


class DocumentProcessor:
    """Extract text from PDF and DOCX files"""

    @staticmethod
    def extract_text(file_obj):
        """
        Extract text from PDF or DOCX file

        Args:
            file_obj: Django UploadedFile object

        Returns:
            dict: {
                'text': str,
                'metadata': {
                    'extraction_method': str,
                    'pages': int or 1,
                    'word_count': int
                }
            }
        """
        file_ext = Path(file_obj.name).suffix.lower()

        # Reset file pointer
        file_obj.seek(0)
        file_content = file_obj.read()
        file_obj.seek(0)

        if file_ext == '.pdf':
            return DocumentProcessor._extract_pdf(file_content, file_obj.name)
        elif file_ext in ['.docx', '.doc']:
            return DocumentProcessor._extract_docx(file_content, file_obj.name)
        elif file_ext == '.txt':
            return DocumentProcessor._extract_txt(file_content, file_obj.name)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    @staticmethod
    def _extract_pdf(file_content, filename):
        """Extract text from PDF"""
        try:
            from pypdf import PdfReader

            pdf_reader = PdfReader(BytesIO(file_content))
            text = ""
            page_count = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page.extract_text()
                except:
                    pass

            word_count = len(text.split())

            return {
                'text': text,
                'metadata': {
                    'extraction_method': 'pypdf',
                    'pages': page_count,
                    'word_count': word_count
                }
            }
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")

    @staticmethod
    def _extract_docx(file_content, filename):
        """Extract text from DOCX file"""
        try:
            from docx import Document

            doc = Document(BytesIO(file_content))
            text = ""

            for para in doc.paragraphs:
                text += para.text + "\n"

            word_count = len(text.split())

            return {
                'text': text,
                'metadata': {
                    'extraction_method': 'python-docx',
                    'pages': 1,
                    'word_count': word_count
                }
            }
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")

    @staticmethod
    def _extract_txt(file_content, filename):
        """Extract text from plain text file"""
        try:
            text = file_content.decode('utf-8')
            word_count = len(text.split())

            return {
                'text': text,
                'metadata': {
                    'extraction_method': 'plain_text',
                    'pages': 1,
                    'word_count': word_count
                }
            }
        except Exception as e:
            raise Exception(f"Text extraction failed: {str(e)}")
