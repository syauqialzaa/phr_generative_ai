import os
import logging
from typing import Dict, List
from llm_interface import LLMInterface

logger = logging.getLogger(__name__)

# Provider Factory
class LLMProviderFactory:
    @staticmethod
    def create_provider(provider_type: str) -> LLMInterface:
        provider_type = provider_type.lower()
        
        if provider_type in ["google", "gemini"]:
            from llm_google import GoogleProvider
            return GoogleProvider()
        elif provider_type == "ollama":
            from llm_ollama import OllamaProvider
            return OllamaProvider()
        else:
            available_providers = ["google", "gemini", "ollama"]
            raise ValueError(f"Unsupported provider type: {provider_type}. Available providers: {available_providers}")

# Provider Manager
class LLMProviderManager:
    def __init__(self):
        self.provider = None
        self.provider_type = None
    
    async def initialize(self, provider_type: str = None) -> bool:
        """Initialize the LLM provider based on environment configuration"""
        if provider_type is None:
            provider_type = os.getenv("LLM_PROVIDER")
        
        try:
            self.provider = LLMProviderFactory.create_provider(provider_type)
            success = await self.provider.initialize()
            
            if success:
                self.provider_type = provider_type
                logger.info(f"LLM Provider '{provider_type}' initialized successfully")
                return True
            else:
                logger.error(f"Failed to initialize LLM Provider '{provider_type}'")
                return False
        except Exception as e:
            logger.error(f"Error creating LLM provider: {e}")
            return False
    
    async def generate_response(self, message: str, history: List[Dict] = None) -> str:
        """Generate response using the configured provider"""
        if not self.provider:
            raise Exception("LLM provider not initialized")
        
        return await self.provider.generate_response(message, history)
    
    def get_provider_info(self) -> Dict:
        """Get information about the current provider"""
        return {
            "provider_type": self.provider_type,
            "initialized": self.provider is not None
        }