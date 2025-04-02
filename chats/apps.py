from django.apps import AppConfig

from chats.views import load_embeddings


class ChatbotConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chats"

    def ready(self):
        load_embeddings()