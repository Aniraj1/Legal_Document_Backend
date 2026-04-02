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


class AskGroqSerializer(serializers.Serializer):
    """
    Serializer for Ask Groq API request (RAG Query)
    """
    field_id = serializers.UUIDField(
        required=True,
        help_text="UUID of the uploaded file/document"
    )
    query = serializers.CharField(
        required=True,
        allow_blank=False,
        max_length=2000,
        min_length=5,
        trim_whitespace=True,
        help_text="User question about the document (5-2000 characters)"
    )
    model = serializers.ChoiceField(
        required=False,
        choices=['llama-3.1-8b-instant', 'llama-3.3-70b-versatile'],
        default='llama-3.1-8b-instant',
        help_text="Groq model to use for response generation"
    )
    chat_history = serializers.ListField(
        required=False,
        default=list,
        child=serializers.DictField(),
        help_text="Optional prior turns for follow-up questions. Format: [{'role': 'user|assistant', 'content': '...'}]"
    )

    def validate_chat_history(self, value):
        """
        Validate history entries and constrain payload size.
        """
        if not isinstance(value, list):
            raise serializers.ValidationError("chat_history must be a list.")

        # Keep payload bounded for latency and token control.
        if len(value) > 20:
            raise serializers.ValidationError("chat_history supports at most 20 messages.")

        validated = []
        for idx, item in enumerate(value):
            if not isinstance(item, dict):
                raise serializers.ValidationError(f"chat_history[{idx}] must be an object.")

            role = item.get('role')
            content = item.get('content')

            if role not in ['user', 'assistant']:
                raise serializers.ValidationError(
                    f"chat_history[{idx}].role must be 'user' or 'assistant'."
                )

            if not isinstance(content, str) or not content.strip():
                raise serializers.ValidationError(
                    f"chat_history[{idx}].content must be a non-empty string."
                )

            # Per-message cap to keep requests predictable.
            if len(content) > 4000:
                raise serializers.ValidationError(
                    f"chat_history[{idx}].content exceeds 4000 characters."
                )

            validated.append({
                'role': role,
                'content': content.strip()
            })

        return validated
