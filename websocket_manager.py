import json
from typing import List, Dict
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.chat_histories: Dict[str, List[Dict]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        if client_id not in self.chat_histories:
            self.chat_histories[client_id] = []
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, client_id: str):
        """Disconnect a WebSocket client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients"""
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    def add_to_history(self, client_id: str, role: str, content: str):
        """Add message to chat history"""
        if client_id not in self.chat_histories:
            self.chat_histories[client_id] = []
        
        self.chat_histories[client_id].append({
            "role": role,
            "content": content,
            "timestamp": json.dumps({"timestamp": "now"})  # Will be replaced with actual timestamp
        })
        
        # Keep only last 20 messages to manage memory
        if len(self.chat_histories[client_id]) > 20:
            self.chat_histories[client_id] = self.chat_histories[client_id][-20:]

    def get_history(self, client_id: str) -> List[Dict]:
        """Get chat history for a client"""
        return self.chat_histories.get(client_id, [])

    def clear_history(self, client_id: str):
        """Clear chat history for a client"""
        if client_id in self.chat_histories:
            self.chat_histories[client_id] = []

    def get_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_chat_sessions": len(self.chat_histories),
            "total_messages": sum(len(history) for history in self.chat_histories.values())
        }