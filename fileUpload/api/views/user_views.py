from django.db import transaction
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle
from globalutils.convert_size import convert_bytes_to_formatted_size

from fileUpload.api.serializer import FileResourceSerializer, AskGroqSerializer
from fileUpload.model.fileresources import FileResource
from fileUpload.services.file_validator import MAX_FILE_SIZE_MB, file_size_validate
from fileUpload.services.document_processor import DocumentProcessor
from fileUpload.services.chunking_service import ChunkingService
from fileUpload.services.vector_service import VectorService
from fileUpload.services.ask_groq_service import AskGroqService
from globalutils.returnobject import project_return


class UploadFileView(GenericAPIView):
    queryset = FileResource.objects.all()
    serializer_class = FileResourceSerializer
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserRateThrottle]

    @extend_schema(tags=["fileUpload"])
    def post(self, request, *args, **kwargs):
        """
        - Validate file (type, size)
        - Extract text from file (in-memory)
        - Intelligently chunk document
        - Uploads chunk to Upstash Vector DB
        - Save metadata to SQLite (ONLY if all above succeed)
        
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
        if not file.content_type == "application/pdf":
            return project_return(
                message="Invalid file type.",
                error="Only PDF files are allowed.",
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        validation = file_size_validate(file.size)
        
        if not validation:
            return project_return(
                message="File size exceeds limit.",
                error=f"File size {convert_bytes_to_formatted_size(file.size)} exceeds maximum limit of {MAX_FILE_SIZE_MB}MB",
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        with transaction.atomic():
            
            file_resource = FileResource.objects.create(
            file_name=file.name,
            file_size=file.size,
            user_id=request.user
            )
            # Step 2: Extract text from file
            doc_data = DocumentProcessor.extract_text(file)

            if not doc_data:
                return project_return(
                    message="Failed to extract text from file.",
                    error="The uploaded file could not be processed. Please ensure it is a valid PDF.",
                    status=status.HTTP_400_BAD_REQUEST,
                )

            text = doc_data['text']
            extraction_metadata = doc_data['metadata']
            
            if not text or not text.strip():
                return project_return(
                    message="Failed to extract text from file.",
                    error="The uploaded file contains no readable text.",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            # Step 3: Chunk document intelligently
            chunks = ChunkingService.chunk_document(
                text=text,
                user_id=request.user.id,
                file_id=file_resource.id,
                file_name=file.name
            )
            
            if not chunks:
                return project_return(
                    message="Failed to chunk document.",
                    error="The extracted text could not be chunked into meaningful sections.",
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            
            # Step 4: Upload chunks to Upstash Vector DB
            vector_service = VectorService()
            upload_result = vector_service.upload_chunks(
                chunks=chunks,
                user_id=str(request.user.id),
                file_id=file_resource.id
            )
            
            if not upload_result['success']:
                return project_return(
                    message="Failed to upload chunks to vector database.",
                    error=f"Vector upload failed: {upload_result['error']}",
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            
            # Step 5: CREATE FileResource in SQLite (ONLY if all above succeeded)
            # Reset file pointer for size reading
            file.seek(0)
            
            
            
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
        

class AskGroqView(GenericAPIView):
    """
    Ask a question about an uploaded document using RAG + Groq API.
    
    Endpoint: POST /api/v1/ask-groq/
    
    This view implements the RAG (Retrieval-Augmented Generation) pattern:
    1. Authenticates user and validates file access
    2. Retrieves top 5 most relevant chunks from Upstash Vector DB
    3. Sends chunks to Groq API with anti-hallucination system prompt
    4. Returns answer with source citations
    """
    queryset = FileResource.objects.all()
    serializer_class = AskGroqSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserRateThrottle]

    @extend_schema(
        tags=["fileUpload"],
        description="Ask a question about an uploaded document using RAG"
    )
    def post(self, request, *args, **kwargs):
        """
        Process user query through RAG pipeline:
        1. Validate request (file_id, query)
        2. Check user has access to file
        3. Retrieve relevant chunks from vector DB
        4. Send to Groq API with grounding prompt
        5. Return answer with citations
        """

        # Step 1: Validate request
        request_obj = self.serializer_class(data=request.data)
        if not request_obj.is_valid():
            return project_return(
                message="Invalid request parameters.",
                error=request_obj.errors,
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        file_id = str(request.data.get('file_id'))
        query = request.data.get('query')
        model = request.data.get('model', 'llama-3.1-8b-instant')
        chat_history = request.data.get('chat_history', [])
        
        # Step 2: Verify file exists and user has access
        file_resource = FileResource.objects.filter(id=file_id, user_id=request.user).first()
        if file_resource is None:
            return project_return(
                    message="File not found.",
                    error="The requested file does not exist.",
                    status=status.HTTP_404_NOT_FOUND,
                )

        
        # Step 3: Process query through RAG service
        rag_service = AskGroqService()
        result = rag_service.process_query(
            user_id=request.user.id,
            file_id=file_id,
            query=query,
            model=model,
            chat_history=chat_history,
        )

        if not result:
            return project_return(
                message="Error processing query.",
                error="An unexpected error occurred while processing your query. Please try again.",
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
            
        # Build response
        data = {
            'answer': result['answer'],
            'sources': result['sources'],
            'confidence': result['confidence'],
            'metadata': result['metadata']
        }
            
        return project_return(
            message="Query processed successfully.",
            data=data,
            status=status.HTTP_200_OK
        )
    

class RemoveUploadedFileView(GenericAPIView):
    """
    (Optional) Endpoint to remove an uploaded file and its chunks from vector DB.
    This can be useful for testing or if users want to delete their data.
    """
    queryset = FileResource.objects.all()
    serializer_class = None
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserRateThrottle]

    @extend_schema(tags=["fileUpload"])
    def delete(self, request, *args, **kwargs):
        """
        Deletes a FileResource and its associated chunks from the vector database.
        Expects 'file_id' in request data to identify which file to delete.
        """
        with transaction.atomic():
            file_id = kwargs.get('file_id')
            if not file_id:
                return project_return(
                    message="Invalid file_id.",
                    error="Please provide the correct file_id.",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            file_resource = FileResource.objects.filter(id=file_id, user_id=request.user).first()
            if not file_resource:
                return project_return(
                    message="File not found.",
                    error="The specified file does not exist or you do not have permission to delete it.",
                    status=status.HTTP_404_NOT_FOUND,
                )
            
            # Delete chunks from vector DB
            vector_service = VectorService()
            delete_result = vector_service.delete_chunks(
                user_id=str(request.user.id),
                file_id=file_id
            )

            if not delete_result.get('success'):
                return project_return(
                    message="Failed to delete file vectors.",
                    error=delete_result.get('error', 'Unknown vector delete error.'),
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            
            # Delete FileResource from SQLite
            file_resource.delete()
            
            return project_return(
                message="File deleted successfully.",
                status=status.HTTP_200_OK
            )
        
class RemoveAllUserUploadedFileView(GenericAPIView):
    """
    (Optional) Endpoint to remove ALL uploaded files and their chunks for the authenticated user.
    Useful for testing or if users want to clear all their data.
    """
    queryset = FileResource.objects.all()
    serializer_class = None
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserRateThrottle]

    @extend_schema(tags=["fileUpload"])
    def delete(self, request, *args, **kwargs):
        """
        Deletes all FileResources and their associated chunks for the authenticated user.
        """
        with transaction.atomic():
            user_id = request.user.id
            file_resources = FileResource.objects.filter(user_id=user_id)
            
            if not file_resources:
                return project_return(
                    message="No files found.",
                    error="You have no uploaded files to delete.",
                    status=status.HTTP_404_NOT_FOUND,
                )
            
            vector_service = VectorService()
            file_ids = list(file_resources.values_list('id', flat=True))
            delete_result = vector_service.delete_all_user_chunks(
                user_id=str(user_id),
                file_ids=file_ids,
            )

            if not delete_result.get('success'):
                return project_return(
                    message="Failed to delete user vectors.",
                    error=delete_result.get('error', 'Unknown vector delete error.'),
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            
            # Delete all FileResources from SQLite
            file_resources.delete()
            
            return project_return(
                message="All your files have been deleted successfully.",
                status=status.HTTP_200_OK
            )