from pathlib import Path


class FileValidator:
    """Validate uploaded files for type and size"""

    ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt'}
    MAX_FILE_SIZE_MB = 50

    @staticmethod
    def validate_file(file_obj):
        """
        Validate uploaded file

        Args:
            file_obj: Django UploadedFile object

        Returns:
            dict: {'valid': bool, 'error': str or None}
        """
        # Check file extension
        ext = Path(file_obj.name).suffix.lower()
        if ext not in FileValidator.ALLOWED_EXTENSIONS:
            return {
                'valid': False,
                'error': f'File type {ext} not allowed. Allowed types: {", ".join(FileValidator.ALLOWED_EXTENSIONS)}'
            }

        # Check file size
        file_size_mb = file_obj.size / (1024 * 1024)
        if file_size_mb > FileValidator.MAX_FILE_SIZE_MB:
            return {
                'valid': False,
                'error': f'File size {file_size_mb:.2f}MB exceeds maximum limit of {FileValidator.MAX_FILE_SIZE_MB}MB'
            }

        return {'valid': True, 'error': None}
