from abc import ABC, abstractmethod
from typing import Dict, List, Any

class LLMInterface(ABC):
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the LLM provider"""
        pass
    
    @abstractmethod
    async def generate_response(self, message: str, history: List[Dict] = None) -> str:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def format_history(self, history: List[Dict]) -> Any:
        """Format chat history for the specific provider"""
        pass