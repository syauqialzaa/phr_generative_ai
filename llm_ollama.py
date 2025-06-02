import os
import logging
from typing import Dict, List
from llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class OllamaProvider(LLMInterface):
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model_name = os.getenv("OLLAMA_MODEL")
        
    async def initialize(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model["name"] for model in models]
                    if not any(self.model_name in name for name in model_names):
                        logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                    logger.info(f"Ollama provider initialized successfully with model: {self.model_name}")
                    return True
                return False
        except ImportError:
            logger.error("httpx package not installed. Install with: pip install httpx")
            return False
        except Exception as e:
            logger.error(f"Error initializing Ollama provider: {e}")
            return False
    
    def format_history(self, history: List[Dict]) -> List[Dict]:
        """Convert generic history to Ollama format"""
        formatted_history = []
        for msg in history:
            role = "user" if msg.get("role") == "user" else "assistant"
            formatted_history.append({
                "role": role,
                "content": msg.get("content", "")
            })
        return formatted_history
    
    async def generate_response(self, message: str, history: List[Dict] = None) -> str:
        try:
            import httpx
            
            messages = self.format_history(history or [])
            messages.append({"role": "user", "content": message})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "No response generated")
                else:
                    raise Exception(f"API error: {response.status_code}")
                    
        except Exception as e:
            raise Exception(f"Provider error: {e}")