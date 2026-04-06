from rest_framework import serializers

from authuser import models


class UserRegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.User
        fields = [
            "id",
            "username",
            "password",
            "email",
            "is_agreement",
        ]



class UserDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.UserDetail
        fields = [
            "first_name",
            "last_name"
        ]


class UserSerializer(serializers.ModelSerializer):
    detail = UserDetailSerializer(read_only=True, source="user_detail")

    class Meta:
        model = models.User
        fields = [
            "id",
            "username",
            "email",
            "is_active",
            "is_user",
            "is_superuser",
            "is_agreement",
            "detail",
        ]





class UserLoginSerializer(serializers.Serializer):
    username = serializers.CharField(required=True)
    password = serializers.CharField(required=True)


class TokenSerializer(serializers.Serializer):
    refresh = serializers.CharField(required=True)


class LoginUserChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)


