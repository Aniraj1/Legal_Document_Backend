import pytest
from decouple import config
from django.urls import reverse

from authuser.api import utils

DEFAULT_PASSWORD = config("SUPER_ADMIN_PASSWORD")


# ------------------- User Login Tests ------------------ #
def test_successful_login(test_user, api_client):
    url = reverse("UserLogin")
    payload = {
        "username": test_user.username,
        "password": DEFAULT_PASSWORD,
    }

    response = api_client.post(url, payload)
    assert response.status_code == 200

def test_login_with_invalid_user(db, api_client):
    url = reverse("UserLogin")
    payload = {"username": "test", "password": DEFAULT_PASSWORD}

    response = api_client.post(url, payload)
    assert response.status_code == 404

def test_login_with_invalid_password(test_user, api_client):
    url = reverse("UserLogin")
    payload = {
        "username": test_user.username,
        "password": f"{DEFAULT_PASSWORD}123",
    }

    response = api_client.post(url, payload)
    assert response.status_code == 401


# ------------------- User Register Tests ------------------ #
def test_register_user_with_valid_data(db, api_client):
    url = reverse("UserRegister")
    payload = {
        "username": "test-123",
        "password": DEFAULT_PASSWORD,
        "is_agreement": True,
        "email": "test@gmail.com",
    }
    response = api_client.post(url, payload, format="json")
    assert response.status_code == 201

def test_register_user_with_no_agreement(db, api_client):
    url = reverse("UserRegister")
    payload = {
        "username": "test-123",
        "password": DEFAULT_PASSWORD,
        "is_agreement": False,
        "email": "test@gmail.com",
    }
    response = api_client.post(url, payload, format="json")
    assert response.status_code == 400

def test_register_user_with_invalid_password_format(db, api_client):
    url = reverse("UserRegister")
    payload = {
        "username": "test-123",
        "password": "test",
        "is_agreement": True,
        "email": "test@gmail.com",
    }
    response = api_client.post(url, payload, format="json")
    assert response.status_code == 400


# ------------------- User Logout Tests ------------------ #
def test_logout_user_with_valid_token(db, login_test_user, api_client):
    url = reverse("UserLogout")
    access_token = login_test_user["access"]
    payload = {"refresh": login_test_user["refresh"]}
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")
    response = api_client.post(url, payload)
    assert response.status_code == 200

def test_logout_user_with_invalid_token(db, login_test_user, api_client):
    url = reverse("UserLogout")
    access_token = login_test_user["access"]
    payload = {"refresh": login_test_user["access"]}
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")
    response = api_client.post(url, payload)
    assert response.status_code == 400

def test_logout_user_without_token(db, login_test_user, api_client):
    url = reverse("UserLogout")
    access_token = login_test_user["access"]
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")
    response = api_client.post(url)
    assert response.status_code == 400


# ------------------- Generate Token From Refresh Tests ------------------ #
def test_generate_token_from_refresh_with_valid_token(db, login_test_user, api_client):
    url = reverse("GenerateTokenFromRefresh")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    payload = {"refresh": login_test_user["refresh"]}
    response = api_client.post(url, payload)
    assert response.status_code == 200

def test_generate_token_from_refresh_with_invalid_token(db, login_test_user, api_client):
    url = reverse("GenerateTokenFromRefresh")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    payload = {"refresh": login_test_user["access"]}
    response = api_client.post(url, payload)
    assert response.status_code == 400

def test_generate_token_from_refresh_without_payload(db, login_test_user, api_client):
    url = reverse("GenerateTokenFromRefresh")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    response = api_client.post(url)
    assert response.status_code == 400


# ------------------- Set User Detail Tests ------------------ #
def test_set_user_detail(db, login_test_user, api_client):  
    url = reverse("UserDetail")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    payload = {
        "first_name": "John",
        "last_name": "Doe",
    }
    response = api_client.post(url, payload)
    assert response.status_code == 201

def test_set_user_detail_with_invalid_token(db, login_test_user, api_client):  
    url = reverse("UserDetail")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('refresh')}"
    )
    payload = {
        "first_name": "John",
        "last_name": "Doe",
    }
    response = api_client.post(url, payload)
    assert response.status_code == 401


def test_set_user_detail_with_no_data(db, login_test_user, api_client):  
    url = reverse("UserDetail")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    payload = {
        "first_name": "",
        "last_name": "s",
    }
    response = api_client.post(url, payload)
    assert response.status_code == 400


# ------------------- Update User Detail Tests ------------------ #
def test_update_user_detail(db, login_test_user, test_user_detail, api_client):  
    url = reverse("UserDetail")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    payload = {
        "first_name": "Jane",
        "last_name": "Smith",
    }
    response = api_client.put(url, payload)

    assert response.status_code == 200

def test_update_user_detail_with_empty_data(db, login_test_user, test_user_detail, api_client):  
    url = reverse("UserDetail")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    payload = {
        "first_name": "",
        "last_name": "",
    }
    response = api_client.put(url, payload)

    assert response.status_code == 400

def test_update_user_detail_with_invalid_token(db, login_test_user, test_user_detail, api_client):  
    url = reverse("UserDetail")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('refresh')}"
    )
    payload = {
        "first_name": "Jane",
        "last_name": "Smith",
    }
    response = api_client.put(url, payload)
    assert response.status_code == 401


# ------------------- Get User Detail Tests ------------------ #
def test_get_user_detail(db, login_test_user, test_user_detail, api_client):  
    url = reverse("UserDetailView")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    response = api_client.get(url)
    assert response.status_code == 200

def test_get_user_detail_with_invalid_token(db, login_test_user, test_user_detail, api_client):  
    url = reverse("UserDetailView")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('refresh')}"
    )
    response = api_client.get(url)
    assert response.status_code == 401


# ------------------- Change Password Tests ------------------ #
def test_change_password(db, login_test_user, api_client):
    url = reverse("LoginUserChangePasswordView")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    payload = {
        "old_password": DEFAULT_PASSWORD,
        "new_password": "NewAdmin@123",
    }
    response = api_client.put(url, payload)
    assert response.status_code == 200

def test_change_password_with_invalid_token(db, login_test_user, api_client):
    url = reverse("LoginUserChangePasswordView")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('refresh')}"
    )
    payload = {
        "old_password": DEFAULT_PASSWORD,
        "new_password": "NewAdmin@123",
    }
    response = api_client.put(url, payload)
    assert response.status_code == 401

def test_change_password_with_invalid_old_password(db, login_test_user, api_client):
    url = reverse("LoginUserChangePasswordView")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    payload = {
        "old_password": f"Old@123",
        "new_password": "NewAdmin@123",
    }
    response = api_client.put(url, payload)
    assert response.status_code == 400

def test_change_password_with_invalid_new_password(db, login_test_user, api_client):
    url = reverse("LoginUserChangePasswordView")
    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {login_test_user.get('access')}"
    )
    payload = {
        "old_password": DEFAULT_PASSWORD,
        "new_password": "new@123",
    }
    response = api_client.put(url, payload)
    assert response.status_code == 400

