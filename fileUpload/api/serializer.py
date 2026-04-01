from rest_framework import serializers
from fileUpload.model.fileresources import FileResource


class FileResourceSerializer(serializers.ModelSerializer):
    """
    Serializer for FileResource model
    """
    file = serializers.FileField(required=True, write_only=True)
    
    class Meta:
        model = FileResource
        fields = ['id', 'file_name', 'file_size', 'user_id', 'file']
        read_only_fields = ['id', 'file_name', 'file_size', 'user_id']
