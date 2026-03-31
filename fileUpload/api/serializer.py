from rest_framework import serializers



class FileResourceSerializer(serializers.Serializer):
    """ 
    - upload file serializer
    """
    file = serializers.FileField(required=True)
    
