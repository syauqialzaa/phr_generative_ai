from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

class ChatMessage(BaseModel):
    question: str
    timestamp: str

class ChatResponse(BaseModel):
    type: str = "response"
    explanation: Optional[str] = None
    message: Optional[str] = None
    visualization: Optional[str] = None
    app_url: Optional[str] = None
    timestamp: str = None

class WellData(BaseModel):
    well: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    economic_limit: Optional[float] = 5

class DCAResult(BaseModel):
    decline_rates: Dict[str, float]
    predictions: Dict[str, List[Dict]]
    timestamp: str

class WellPerformance(BaseModel):
    well: str
    avg_production: float
    decline_rate: float
    data_points: int