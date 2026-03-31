from decouple import config
from django.urls import path

from fileUpload.api.views import user_views

urlpatterns = [
    path(
        "upload/", user_views.UploadFileView.as_view(), name="UploadFileView"    
    ),
]
