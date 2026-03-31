import uuid

from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _
from authuser.model.user import User



class FileResource(models.Model):
    """
    Resources where uploaded files detail are present
    """
    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, db_column="ID"
    )
    file_name = models.TextField(db_column="FILE_NAME", null=True, blank=True)
    file_size = models.CharField(db_column="FILE_SIZE", null=True, blank=True)
    user_id = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        db_column="USER_ID",
        related_name="file_upload_user",
    )

    class Meta:
        db_table = "POC_FILE_RESOURCE"

    def __str__(self):
        return str(self.id)