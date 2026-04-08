from decouple import config
from django.urls import path

from authuser.api.views import user_views

urlpatterns = [
    path(
        f"{config('PROJECT_NAME')}/user/register/",
        user_views.UserRegister.as_view(),
        name="UserRegister",
    ),
    path(
        f"{config('PROJECT_NAME')}/user/login/",
        user_views.UserLogin.as_view(),
        name="UserLogin",
    ),
    path(
        f"{config('PROJECT_NAME')}/user/logout/",
        user_views.UserLogout.as_view(),
        name="UserLogout",
    ),
    path(
        f"{config('PROJECT_NAME')}/refresh-token/",
        user_views.GenerateTokenFromRefresh.as_view(),
        name="GenerateTokenFromRefresh",
    ),
    path(
        "logged-in-user/change-password/",
        user_views.LoginUserChangePasswordView.as_view(),
        name="LoginUserChangePasswordView",
    ),
    path("user-detail/", user_views.UserDetail.as_view(), name="UserDetail"),
    path("user-info/", user_views.UserDetailView.as_view(), name="UserDetailView"),
]
