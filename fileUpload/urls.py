from django.urls import include, path

urlpatterns = [
    path("v1/", include("fileUpload.api.urls.user_urls")),
]