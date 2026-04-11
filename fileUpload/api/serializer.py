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


class FileResourceListSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileResource
        fields = ['id', 'file_name']


class AskGroqSerializer(serializers.Serializer):
    """
    Serializer for Ask Groq API request (RAG Query)
    """
    file_id = serializers.UUIDField(
        required=True,
        help_text="UUID of the uploaded file/document"
    )
    query = serializers.CharField(
        required=True,
        allow_blank=False,
        max_length=2000,
        min_length=1,
        trim_whitespace=True,
        help_text="User question about the document (1-2000 characters)"
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
        help_text="Optional prior turns for follow-up questions. Format: [{'role': 'user|assistant', 'content': '...'}]. If more than 20 messages are provided, only the latest 20 are used."
    )

    MAX_CHAT_HISTORY_MESSAGES = 20

    def validate_chat_history(self, value):
        """
        Validate history entries and constrain payload size.
        """
        if not isinstance(value, list):
            raise serializers.ValidationError("chat_history must be a list.")

        # Keep payload bounded for latency and token control.
        # If the client sends more, keep only the latest messages instead of failing the request.
        if len(value) > self.MAX_CHAT_HISTORY_MESSAGES:
            value = value[-self.MAX_CHAT_HISTORY_MESSAGES:]

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

    def validate_query(self, value):
        """Allow short prompts, but reject inputs with no alphanumeric signal."""
        cleaned = (value or "").strip()
        if not cleaned:
            raise serializers.ValidationError("query must not be empty.")

        if not any(ch.isalnum() for ch in cleaned):
            raise serializers.ValidationError("query must contain letters or numbers.")

        return cleaned
