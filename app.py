import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import logging

# Import LLM provider manager
from llm_providers import LLMProviderManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM Provider Manager
provider_manager = LLMProviderManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler untuk startup dan shutdown"""
    # Startup events
    logger.info("Starting up PHR Generative AI Chat Server...")
    success = await provider_manager.initialize()
    if not success:
        logger.error("Failed to initialize AI service")
    else:
        provider_info = provider_manager.get_provider_info()
        logger.info(f"AI service ready: {provider_info}")
    
    yield  # Aplikasi berjalan di sini
    
    # Shutdown events (optional)
    logger.info("Shutting down PHR Generative AI Chat Server...")
    # Tambahkan cleanup code di sini jika diperlukan
    # await provider_manager.cleanup()

# Initialize FastAPI app dengan lifespan
app = FastAPI(
    title="PHR Generative AI Chat Server", 
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    question: str
    timestamp: str

class ChatResponse(BaseModel):
    type: str = "response"
    explanation: Optional[str] = None
    message: Optional[str] = None
    timestamp: str = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.chat_histories: Dict[str, List[Dict]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if client_id not in self.chat_histories:
            self.chat_histories[client_id] = []
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, client_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def add_to_history(self, client_id: str, role: str, content: str):
        if client_id not in self.chat_histories:
            self.chat_histories[client_id] = []
        
        self.chat_histories[client_id].append({
            "role": role,
            "content": content
        })
        
        # Keep only last 20 messages to manage memory
        if len(self.chat_histories[client_id]) > 20:
            self.chat_histories[client_id] = self.chat_histories[client_id][-20:]

    def get_history(self, client_id: str) -> List[Dict]:
        return self.chat_histories.get(client_id, [])

    def clear_history(self, client_id: str):
        if client_id in self.chat_histories:
            self.chat_histories[client_id] = []

manager = ConnectionManager()

# Generate response using configured LLM provider
async def generate_response(question: str, client_id: str) -> dict:
    try:
        if not provider_manager.provider:
            return {
                "type": "error",
                "message": "AI service not available",
                "timestamp": datetime.now().isoformat()
            }

        # Get chat history for context
        history = manager.get_history(client_id)
        
        # Generate response
        response_text = await provider_manager.generate_response(question, history)
        
        # Add to history
        manager.add_to_history(client_id, "user", question)
        manager.add_to_history(client_id, "assistant", response_text)
        
        return {
            "type": "response",
            "explanation": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "type": "error",
            "message": f"Error generating response: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Index.html not found</h1>", status_code=404)

# Health check endpoint
@app.get("/health")
async def health_check():
    provider_info = provider_manager.get_provider_info()
    return {
        "status": "healthy" if provider_info["initialized"] else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ai_service": provider_info
    }

# Get provider information
@app.get("/api/provider-info")
async def get_provider_info():
    return provider_manager.get_provider_info()

# REST API endpoint for chat
@app.post("/api/query")
async def process_query(message: ChatMessage):
    try:
        client_id = f"rest_client_{datetime.now().timestamp()}"
        
        logger.info(f"Processing query from {client_id}: {message.question}")
        
        response = await generate_response(message.question, client_id)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing REST query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"ws_client_{datetime.now().timestamp()}"
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        provider_info = provider_manager.get_provider_info()
        provider_name = provider_info.get("provider_type", "AI").title()
        
        welcome_message = {
            "type": "response",
            "explanation": f"Selamat datang di PHR Generative AI (powered by {provider_name})! Saya siap membantu Anda. Silakan ajukan pertanyaan dalam bahasa apapun.",
            "timestamp": datetime.now().isoformat()
        }
        await manager.send_personal_message(welcome_message, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            logger.info(f"Received WebSocket message from {client_id}: {message_data.get('question', '')}")
            
            # Process the question
            response = await generate_response(
                message_data.get("question", ""), 
                client_id
            )
            
            # Send response back to client
            await manager.send_personal_message(response, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        error_response = {
            "type": "error",
            "message": "Terjadi kesalahan dalam koneksi",
            "timestamp": datetime.now().isoformat()
        }
        try:
            await manager.send_personal_message(error_response, websocket)
        except:
            pass
        manager.disconnect(websocket, client_id)

# Clear chat history endpoint
@app.delete("/api/clear-history/{client_id}")
async def clear_chat_history(client_id: str):
    try:
        manager.clear_history(client_id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get chat statistics
@app.get("/api/stats")
async def get_stats():
    provider_info = provider_manager.get_provider_info()
    return {
        "active_connections": len(manager.active_connections),
        "total_chat_sessions": len(manager.chat_histories),
        "ai_service": provider_info,
        "timestamp": datetime.now().isoformat()
    }

# Switch provider endpoint (for development/testing)
@app.post("/api/switch-provider/{provider_type}")
async def switch_provider(provider_type: str):
    try:
        success = await provider_manager.initialize(provider_type)
        if success:
            return {"message": f"Successfully switched to {provider_type} provider"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to initialize {provider_type} provider")
    except Exception as e:
        logger.error(f"Error switching provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting PHR Generative AI Chat Server on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )