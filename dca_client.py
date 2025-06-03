import aiohttp
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DCAApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_wells(self) -> List[str]:
        """Get list of available wells"""
        try:
            async with self.session.get(f"{self.base_url}/get_wells") as response:
                data = await response.json()
                return data.get("wells", [])
        except Exception as e:
            logger.error(f"Error getting wells: {e}")
            return []
    
    async def get_history(self, well: str = None, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get historical production data"""
        try:
            payload = {}
            if well:
                payload["well"] = well
            if start_date:
                payload["start_date"] = start_date
            if end_date:
                payload["end_date"] = end_date
            
            async with self.session.post(f"{self.base_url}/get_history", json=payload) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
    
    async def automatic_dca(self, well: str, selected_data: List[Dict] = None) -> Dict:
        """Perform automatic DCA analysis"""
        try:
            payload = {"well": well}
            if selected_data:
                payload["selected_data"] = selected_data
            
            async with self.session.post(f"{self.base_url}/automatic_dca", json=payload) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error performing DCA: {e}")
            return {}
    
    async def predict_production(self, well: str, economic_limit: float = 5, selected_data: Dict = None) -> Dict:
        """Predict future production"""
        try:
            payload = {"well": well, "economic_limit": economic_limit}
            if selected_data:
                payload["selected_data"] = selected_data
            
            async with self.session.post(f"{self.base_url}/predict_production", json=payload) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error predicting production: {e}")
            return {}
    
    async def predict_ml(self, elr: float = 10) -> Dict:
        """Get ML-based predictions"""
        try:
            payload = {"elr": elr}
            async with self.session.post(f"{self.base_url}/predict_ml", json=payload) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting ML predictions: {e}")
            return {}