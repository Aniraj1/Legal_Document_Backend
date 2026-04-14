import pytest
from pytest_factoryboy import register
from rest_framework.test import APIClient
from authuser.api.utils import get_tokens_for_user
from authuser.factories import UserFactory
from authuser.model.user import User
from fileUpload.factories import FileResourceFactory
from fileUpload.model.fileresources import FileResource

register(UserFactory)
register(FileResourceFactory)


@pytest.fixture()
def api_client():
    return APIClient()


@pytest.fixture()
def test_user(db, user_factory):
    user_from_factory = user_factory.build()
    user = User.objects.create(
        username=user_from_factory.username,
        password=user_from_factory.password,
        email=user_from_factory.email,
        is_superuser=True,
        is_agreement=True,
    )
    return user

@pytest.fixture()
def test_file_resource(db, test_user, file_resource_factory):
    file_from_factory = file_resource_factory.build()
    fileResource = FileResource.objects.create(
        file_name=file_from_factory.file_name,
        file_size=file_from_factory.file_size,
        user_id=test_user, 
    )
    return fileResource


@pytest.fixture()
def login_test_user(db, test_user):
    return get_tokens_for_user(test_user)
