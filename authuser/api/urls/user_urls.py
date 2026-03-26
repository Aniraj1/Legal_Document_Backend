from decouple import config
from django.urls import path

from authuser.api.views import user_views

urlpatterns = [
    path(
        f"{config('PROJECT_NAME')}/user/register/",
        user_views.UserRegister.as_view(),
        name="UserRegister",
    ),
]
