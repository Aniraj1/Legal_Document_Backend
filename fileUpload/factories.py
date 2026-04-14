import factory
from faker import Faker
from fileUpload.model.fileresources import FileResource
from authuser.factories import UserFactory

fake = Faker()


class FileResourceFactory(factory.Factory):
    class Meta:
        model = FileResource

    file_name = fake.file_name()
    file_size = "5000"
    user_id = factory.SubFactory(UserFactory)