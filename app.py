import json
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import logging

# Import custom modules
from models import ChatMessage, ChatResponse
from config import SERVER_HOST, SERVER_PORT, APP_NAME, APP_VERSION
from unified_assistant import UnifiedAssistant
from websocket_manager import ConnectionManager
from llm_providers import LLMProviderManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM Provider Manager
provider_manager = LLMProviderManager()

# Initialize Connection Manager
manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup events
    logger.info(f"Starting up {APP_NAME} v{APP_VERSION}...")
    
    # Initialize LLM
    success = await provider_manager.initialize()
    if not success:
        logger.error("Failed to initialize AI service")
    else:
        provider_info = provider_manager.get_provider_info()
        logger.info(f"AI service ready: {provider_info}")
    
    # Initialize Unified Assistant (handles both DCA and Wellbore)
    app.state.unified_assistant = UnifiedAssistant()
    logger.info("DCA and Wellbore Assistant initialized successfully")
    
    yield  # Application runs here
    
    # Shutdown events
    logger.info(f"Shutting down {APP_NAME}...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title=APP_NAME, 
    version=APP_VERSION,
    description="Advanced AI assistant for DCA analysis and wellbore visualization",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

async def generate_response(question: str, client_id: str, app_state) -> dict:
    """Generate response using Unified Assistant"""
    try:
        # Get chat history for context
        history = manager.get_history(client_id)
        
        # Process with Unified Assistant (handles both DCA and Wellbore)
        response = await app_state.unified_assistant.process_query(question, history)
        
        # Add to history
        manager.add_to_history(client_id, "user", question)
        if response.get("explanation"):
            manager.add_to_history(client_id, "assistant", response["explanation"])
        
        return response
        
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
    """Serve the main page"""
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Index.html not found</h1>", status_code=404)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    provider_info = provider_manager.get_provider_info()
    return {
        "status": "healthy" if provider_info["initialized"] else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ai_service": provider_info,
        "dca_integration": True,
        "wellbore_integration": True,
        "version": APP_VERSION
    }

@app.get("/api/provider-info")
async def get_provider_info():
    """Get AI provider information"""
    return provider_manager.get_provider_info()

@app.post("/api/query")
async def process_query(message: ChatMessage):
    """REST API endpoint for processing queries"""
    try:
        client_id = f"rest_client_{datetime.now().timestamp()}"
        
        logger.info(f"Processing query from {client_id}: {message.question}")
        
        response = await generate_response(message.question, client_id, app.state)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing REST query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    client_id = f"ws_client_{datetime.now().timestamp()}"
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        provider_info = provider_manager.get_provider_info()
        provider_name = provider_info.get("provider_type", "AI").title()
        
        welcome_message = {
            "type": "response",
            "explanation": f"Selamat datang di PHR Generative AI dengan DCA & Wellbore Integration (powered by {provider_name})! 🚀\n\n"
                          f"✅ **LAYANAN TERSEDIA:**\n"
                          f"🔬 **DCA Analysis** - Decline curve analysis dan prediksi produksi\n"
                          f"🏗️ **Wellbore Diagrams** - Visualisasi dan analisis struktur sumur\n"
                          f"🤖 **Machine Learning** - Prediksi AI untuk pola produksi kompleks\n\n"
                          f"💡 **CONTOH PERTANYAAN:**\n"
                          f"• \"Tampilkan diagram wellbore untuk sumur PEB000026D1\"\n"
                          f"• \"Analisis DCA untuk sumur PKU00001-01\"\n"
                          f"• \"Show wellbore components dengan ESP system\"\n"
                          f"• \"Prediksi ML produksi dengan ELR 10 BOPD\"\n\n"
                          f"Silakan ajukan pertanyaan dalam bahasa Indonesia atau English! 🌐",
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
                client_id,
                app.state
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

@app.delete("/api/clear-history/{client_id}")
async def clear_chat_history(client_id: str):
    """Clear chat history for a specific client"""
    try:
        manager.clear_history(client_id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get application statistics"""
    provider_info = provider_manager.get_provider_info()
    connection_stats = manager.get_stats()
    
    return {
        **connection_stats,
        "ai_service": provider_info,
        "dca_integration": True,
        "wellbore_integration": True,
        "version": APP_VERSION,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/chat-history/{client_id}")
async def get_chat_history(client_id: str):
    """Get chat history for a specific client"""
    try:
        history = manager.get_history(client_id)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"Starting {APP_NAME} v{APP_VERSION} on {SERVER_HOST}:{SERVER_PORT}")
    
    uvicorn.run(
        "app:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
        log_level="info"
    )