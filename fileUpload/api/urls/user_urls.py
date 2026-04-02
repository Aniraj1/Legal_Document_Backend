from decouple import config
from django.urls import path

from fileUpload.api.views import user_views

urlpatterns = [
    path(
        "upload/", user_views.UploadFileView.as_view(), name="UploadFileView"    
    ),
    path(
        "ask-groq/", user_views.AskGroqView.as_view(), name="AskGroqView"
    ),
]
