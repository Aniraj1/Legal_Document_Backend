import pytest
from django.urls import reverse

from authuser.api import utils

# ------------------- File Upload API Endpoint Tests ------------------ #
# def test_upload_file_with_valid_data(api_client, login_test_user):
#     url = reverse("UploadFileView")
#     api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
#     payload = {
#         "file": open("docs/submission_aayush Annotated.pdf", "rb"),
#     }
#     response = api_client.post(url, payload, format="multipart")
#     print(response.data)
#     assert response.status_code == 201

def test_upload_file_with_invalid_file_type(api_client, login_test_user):
    url = reverse("UploadFileView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
    payload = {
        "file": open("./README.md", "rb"),
    }
    response = api_client.post(url, payload, format="multipart")
    assert response.status_code == 400

def test_upload_file_with_invalid_token(api_client, login_test_user):
    url = reverse("UploadFileView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['refresh']}")
    payload = {
        "file": open("./README.md", "rb"),
    }
    response = api_client.post(url, payload, format="multipart")
    assert response.status_code == 401


# -------------------- List User Files API Endpoint Tests ------------------ #
def test_list_user_files(api_client, test_file_resource, login_test_user):
    url = reverse("ListUserFilesView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
    response = api_client.get(url)
    assert response.status_code == 200

def test_list_user_files_with_invalid_token(api_client, test_file_resource, login_test_user):
    url = reverse("ListUserFilesView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['refresh']}")
    response = api_client.get(url)
    assert response.status_code == 401

# -------------------- Ask Groq API Endpoint Tests ------------------ #
def test_ask_groq_with_valid_query(api_client, test_file_resource, login_test_user):
    url = reverse("AskGroqView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
    payload = {
        "file_id": test_file_resource.id,
        "query": "What is the name of the file?",
        "model": "llama-3.1-8b-instant",
    }
    response = api_client.post(url, payload)
    assert response.status_code == 200

def test_ask_groq_with_invalid_token(api_client, test_file_resource, login_test_user):
    url = reverse("AskGroqView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['refresh']}")
    payload = {
        "file_id": test_file_resource.id,
        "query": "What is the name of the file?",
        "model": "llama-3.1-8b-instant",
    }
    response = api_client.post(url, payload)
    assert response.status_code == 401

def test_ask_groq_with_invalid_model(api_client, test_file_resource, login_test_user):
    url = reverse("AskGroqView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
    payload = {
        "file_id": test_file_resource.id,
        "query": "What is the name of the file?",
        "model": "invalid_model",
    }
    response = api_client.post(url, payload)
    print(response.data)
    assert response.status_code == 400


# ----------------------- Remove Uploaded File API Endpoint Tests ------------------ #
def test_remove_uploaded_file(api_client, test_file_resource, login_test_user):
    url = reverse("RemoveUploadedFileView", args=[test_file_resource.id])
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
    response = api_client.delete(url)
    assert response.status_code == 200

def test_remove_uploaded_file_with_invalid_token(api_client, test_file_resource, login_test_user):
    url = reverse("RemoveUploadedFileView", args=[test_file_resource.id])
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['refresh']}")
    response = api_client.delete(url)
    assert response.status_code == 401

def test_remove_uploaded_file_with_invalid_file_id(api_client, test_file_resource, login_test_user):
    url = reverse("RemoveUploadedFileView", args=[str("3fa85f64-5717-4562-b3fc-2c963f66afa6")])
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
    response = api_client.delete(url)
    assert response.status_code == 404


# ----------------------- Remove All User Uploaded Files API Endpoint Tests ------------------ #
def test_remove_all_user_uploaded_files(api_client, test_file_resource, login_test_user):
    url = reverse("RemoveAllUserUploadedFileView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
    response = api_client.delete(url)
    assert response.status_code == 200

def test_remove_all_user_uploaded_files_with_invalid_token(api_client, test_file_resource, login_test_user):
    url = reverse("RemoveAllUserUploadedFileView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['refresh']}")
    response = api_client.delete(url)
    assert response.status_code == 401


# ----------------------- Personal Analytics API Endpoint Tests ------------------ #
def test_get_personal_analytics(api_client, test_file_resource, login_test_user):
    url = reverse("PersonalAnalyticsView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
    response = api_client.get(url)
    assert response.status_code == 200

def test_get_personal_analytics_with_invalid_token(api_client, test_file_resource, login_test_user):
    url = reverse("PersonalAnalyticsView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['refresh']}")
    response = api_client.get(url)
    assert response.status_code == 401


# ------------------------ Clear Personal Analytics API Endpoint Tests ------------------ #
def test_clear_personal_analytics(api_client, test_file_resource, login_test_user):
    url = reverse("PersonalAnalyticsView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['access']}")
    response = api_client.delete(url)
    assert response.status_code == 200

def test_clear_personal_analytics_with_invalid_token(api_client, test_file_resource, login_test_user):
    url = reverse("PersonalAnalyticsView")
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_test_user['refresh']}")
    response = api_client.delete(url)
    assert response.status_code == 401