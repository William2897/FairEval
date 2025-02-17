from channels.generic.websocket import AsyncWebsocketConsumer
import json

class ProfessorUpdateConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.professor_id = self.scope['url_route']['kwargs']['professor_id']
        self.group_name = f"professor_{self.professor_id}"

        # Join professor group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave professor group
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def professor_update(self, event):
        # Send professor update to WebSocket
        await self.send(text_data=json.dumps({
            "type": "professor.update",
            "professor_id": event["professor_id"]
        }))