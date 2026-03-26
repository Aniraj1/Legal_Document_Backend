from django.urls import include, path

urlpatterns = [
    path("v1/", include("authuser.api.urls.user_urls")),
]