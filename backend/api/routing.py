from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/professors/(?P<professor_id>\w+)/$', consumers.ProfessorUpdateConsumer.as_asgi()),
]