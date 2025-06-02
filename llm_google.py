import os
import asyncio
import logging
from typing import Dict, List
from llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class GoogleProvider(LLMInterface):
    def __init__(self):
        self.model = None
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL")
        
    async def initialize(self) -> bool:
        try:
            if not self.api_key:
                logger.error("Google API key not found in environment variables")
                return False
                
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Google provider initialized successfully with model: {self.model_name}")
            return True
        except ImportError:
            logger.error("google-generativeai package not installed. Install with: pip install google-generativeai")
            return False
        except Exception as e:
            logger.error(f"Error initializing Google provider: {e}")
            return False
    
    def format_history(self, history: List[Dict]) -> List[Dict]:
        """Convert generic history to Google format"""
        formatted_history = []
        for msg in history:
            if msg.get("role") == "user":
                formatted_history.append({"role": "user", "parts": [msg.get("content", "")]})
            elif msg.get("role") == "assistant":
                formatted_history.append({"role": "model", "parts": [msg.get("content", "")]})
        return formatted_history
    
    async def generate_response(self, message: str, history: List[Dict] = None) -> str:
        try:
            if not self.model:
                raise Exception("Model not initialized")
            
            formatted_history = self.format_history(history or [])
            chat = self.model.start_chat(history=formatted_history)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, chat.send_message, message
            )
            return response.text
        except Exception as e:
            raise Exception(f"Provider error: {e}")