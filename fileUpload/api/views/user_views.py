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
from globalutils.returnobject import project_return


class UploadFileView(GenericAPIView):
    queryset = None
    serializer_class = FileResourceSerializer
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserRateThrottle]

    @extend_schema(tags=["fileUpload"])
    def post(self, request, *args, **kwargs):
        request_obj = self.serializer_class(data=request.FILES)
        if not request_obj.is_valid():
            return project_return(
                message="Invalid data.",
                error=request_obj.errors,
                status=status.HTTP_400_BAD_REQUEST,
            )
        file = request.FILES["file"]

        if not file:
            return project_return(
                message="No file provided.",
                error="File is required.",
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        file_resource = FileResource.objects.create(
            file_name=file.name,
            file_size=file.size,
            user_id=request.user,
        )

        
        data = {
            "id": file_resource.id,
            "file_name": file_resource.file_name,
            "file_size": convert_bytes_to_formatted_size(file_resource.file_size),
            "user_id": file_resource.user_id_id,
        }
        return project_return(
            message="Uploaded Successfully.", data=data, status=status.HTTP_200_OK
        )