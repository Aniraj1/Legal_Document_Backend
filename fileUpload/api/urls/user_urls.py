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
    path(
        "remove-file/<str:file_id>/", user_views.RemoveUploadedFileView.as_view(), name="RemoveUploadedFileView"
    ),
    path(
        "remove-files/", user_views.RemoveAllUserUploadedFileView.as_view(), name="RemoveAllUserUploadedFileView"
    ),
    path(
        "get-files/", user_views.ListUserFilesView.as_view(), name="ListUserFilesView"
    ),
    path(
        "analytics/me/", user_views.PersonalAnalyticsView.as_view(), name="PersonalAnalyticsView"
    ),
]
