from django.db import transaction
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle
from globalutils.convert_size import convert_bytes_to_formatted_size

from fileUpload.api.serializer import FileResourceSerializer
from fileUpload.model.fileresources import FileResource
from fileUpload.services.file_validator import FileValidator
from fileUpload.services.document_processor import DocumentProcessor
from fileUpload.services.chunking_service import ChunkingService
from fileUpload.services.vector_service import VectorService
from globalutils.returnobject import project_return


class UploadFileView(GenericAPIView):
    queryset = FileResource.objects.all()
    serializer_class = FileResourceSerializer
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserRateThrottle]

    @extend_schema(tags=["fileUpload"])
    @transaction.atomic()
    def post(self, request, *args, **kwargs):
        """
        Atomic file upload with processing:
        1. Validate file (type, size)
        2. Extract text from file (in-memory)
        3. Intelligently chunk document
        4. Upload chunks to Upstash Vector DB
        5. Save metadata to SQLite (ONLY if all above succeed)
        
        If ANY step fails → Exception raised → Automatic rollback
        """
        
        # Step 1: Get file and validate
        file = request.FILES.get("file")
        if not file:
            return project_return(
                message="No file provided.",
                error="File is required.",
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        validation = FileValidator.validate_file(file)
        if not validation['valid']:
            return project_return(
                message="Invalid file.",
                error=validation['error'],
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        try:
            # Step 2: Extract text from file
            doc_data = DocumentProcessor.extract_text(file)
            text = doc_data['text']
            extraction_metadata = doc_data['metadata']
            
            if not text or not text.strip():
                raise ValueError("File contains no readable text")
            
            # Step 3: Chunk document intelligently
            chunks = ChunkingService.chunk_document(
                text=text,
                user_id=request.user.id,
                file_id=None,  # Will use auto-generated UUID
                file_name=file.name
            )
            
            if not chunks:
                raise ValueError("Document could not be chunked into meaningful sections")
            
            # Step 4: Upload chunks to Upstash Vector DB
            vector_service = VectorService()
            upload_result = vector_service.upload_chunks(
                chunks=chunks,
                user_id=str(request.user.id),
                file_id=None  # Will use auto-generated UUID
            )
            
            if not upload_result['success']:
                raise Exception(f"Vector upload failed: {upload_result['error']}")
            
            # Step 5: CREATE FileResource in SQLite (ONLY if all above succeeded)
            # Reset file pointer for size reading
            file.seek(0)
            
            file_resource = FileResource.objects.create(
                file_name=file.name,
                file_size=file.size,
                user_id=request.user
            )
            
            # Return success response
            data = {
                "id": str(file_resource.id),
                "file_name": file_resource.file_name,
                "file_size": convert_bytes_to_formatted_size(file_resource.file_size),
                "user_id": str(file_resource.user_id_id),
                "chunks_created": upload_result['chunk_count'],
                "extraction_metadata": extraction_metadata,
            }
            
            return project_return(
                message="File uploaded, processed, and stored successfully.",
                data=data,
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            # ANY exception here triggers automatic rollback
            # No FileResource saved, transaction rolled back completely
            return project_return(
                message="File processing failed.",
                error=str(e),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )