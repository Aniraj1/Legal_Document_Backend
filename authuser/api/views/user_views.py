
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework.throttling import AnonRateThrottle


from authuser import models
from authuser.api import serializer, utils
from globalutils.returnobject import project_return


class UserRegister(GenericAPIView):
    """
    - User register using username, email, password and phone(optional)
    """

    queryset = models.User
    serializer_class = serializer.UserRegisterSerializer
    throttle_classes = [AnonRateThrottle]

    @extend_schema(tags=["authuser"])
    def post(self, request, *args, **kwargs):
        user_obj = self.serializer_class(data=request.data)

        if user_obj.is_valid():
            try:
                validate_password(password=request.data.get("password"))
            except ValidationError as e:
                return project_return(
                    message="Not created.",
                    error=e.args,
                    status=status.HTTP_400_BAD_REQUEST,
                )
            check_email = models.User.objects.filter(
                email=request.data.get("email")
            )
            if check_email:
                return project_return(
                    message="Not created.",
                    error="User with this email already exists.",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            if request.data.get("is_agreement") is not True:
                return project_return(
                    message="Not created.",
                    error="User must agree to the terms and conditions.",
                    status=status.HTTP_400_BAD_REQUEST,
                )


            user_obj.save(
                password=make_password(password=request.data.get("password")),
            )

            return project_return(
                message="Successfully created.",
                data=utils.get_tokens_for_user(
                    models.User.objects.get(id=user_obj.data.get("id"))
                ),
                status=status.HTTP_201_CREATED,
            )
        return project_return(
            message="Not created.",
            error=user_obj.errors,
            status=status.HTTP_400_BAD_REQUEST,
        )


class UserLogin(GenericAPIView):
    """
    - User login using username and password and return access and refresh token
    """

    queryset = models.User
    serializer_class = serializer.UserLoginSerializer
    throttle_classes = [AnonRateThrottle]

    @extend_schema(tags=["authuser"])
    def post(self, request, *args, **kwargs):
        request_obj = self.serializer_class(data=request.data)
        if not request_obj.is_valid():
            return project_return(
                message="Not logged in.",
                error=request_obj.errors,
                status=status.HTTP_401_UNAUTHORIZED,
            )

        user = authenticate(
            username=request.data.get("username"),
            password=request.data.get("password"),
        )

        return_credentials = utils.get_tokens_for_user(user) if user else None


        print("working????")
        check_username = models.User.objects.filter(
            username=request.data.get("username")
        ).first()

        if not check_username:
            return project_return(
                message="User does not exist.",
                status=status.HTTP_404_NOT_FOUND,
            )


        return project_return(
            message="Successfully logged in." if user else "Not logged in.",
            data= return_credentials,
            status=status.HTTP_200_OK if user else status.HTTP_401_UNAUTHORIZED,
        )