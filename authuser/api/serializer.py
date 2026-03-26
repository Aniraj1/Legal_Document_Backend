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
            "last_name",
            "street_address",
            "city",
            "state",
            "zip_code",
            "country",
        ]


class UserSerializer(serializers.ModelSerializer):
    detail = UserDetailSerializer(read_only=True, source="user_detail")

    class Meta:
        model = models.User
        fields = [
            "id",
            "username",
            "email",
            "phone",
            "is_active",
            "is_user",
            "is_superuser",
            "created_at",
            "updated_at",
            "roles",
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


