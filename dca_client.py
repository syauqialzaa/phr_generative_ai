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
        """Get list of available wells with proper error handling"""
        try:
            async with self.session.get(f"{self.base_url}/get_wells") as response:
                if response.status != 200:
                    logger.error(f"Get wells failed with status {response.status}")
                    return []
                    
                data = await response.json()
                wells = data.get("wells", [])
                logger.info(f"Retrieved {len(wells)} wells from API")
                return wells
                
        except Exception as e:
            logger.error(f"Error getting wells: {e}")
            return []
    
    async def get_history(self, well: str = None, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get historical production data with enhanced parameter handling"""
        try:
            # Build payload with proper parameter validation
            payload = {}
            
            if well:
                payload["well"] = well
                logger.info(f"History request for well: {well}")
            
            if start_date:
                payload["start_date"] = start_date
                logger.info(f"Start date filter: {start_date}")
            
            if end_date:
                payload["end_date"] = end_date
                logger.info(f"End date filter: {end_date}")
            
            # Log the complete payload for debugging
            logger.info(f"Sending history request with payload: {payload}")
            
            async with self.session.post(
                f"{self.base_url}/get_history", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Get history failed with status {response.status}: {error_text}")
                    return []
                
                data = await response.json()
                
                # Handle different response formats
                if isinstance(data, list):
                    result = data
                elif isinstance(data, dict) and 'data' in data:
                    result = data['data']
                elif isinstance(data, dict) and 'history' in data:
                    result = data['history']
                else:
                    logger.warning(f"Unexpected response format: {type(data)}")
                    result = []
                
                logger.info(f"Retrieved {len(result)} history records")
                return result
                
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
    
    async def automatic_dca(self, well: str, selected_data: List[Dict] = None) -> Dict:
        """Perform automatic DCA analysis with proper parameter handling"""
        try:
            # Build payload with validation
            payload = {"well": well}
            
            if selected_data:
                payload["selected_data"] = selected_data
                logger.info(f"DCA analysis with {len(selected_data)} selected data points")
            else:
                logger.info(f"DCA analysis for well {well} with all available data")
            
            logger.info(f"Sending DCA request with payload keys: {list(payload.keys())}")
            
            async with self.session.post(
                f"{self.base_url}/automatic_dca", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"DCA analysis failed with status {response.status}: {error_text}")
                    return {"error": f"API returned status {response.status}: {error_text}"}
                
                data = await response.json()
                
                # Validate response structure
                if not isinstance(data, dict):
                    logger.error(f"Invalid DCA response type: {type(data)}")
                    return {"error": "Invalid response format"}
                
                # Check for API error responses
                if 'error' in data:
                    logger.error(f"DCA API returned error: {data['error']}")
                    return data
                
                # Validate required fields
                required_fields = ['DeclineRate', 'Exponential', 'Harmonic', 'Hyperbolic']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    logger.warning(f"DCA response missing fields: {missing_fields}")
                
                logger.info(f"DCA analysis successful, response keys: {list(data.keys())}")
                return data
                
        except Exception as e:
            logger.error(f"Error performing DCA: {e}")
            return {"error": str(e)}
    
    async def predict_production(self, well: str, economic_limit: float = 5, selected_data: Dict = None) -> Dict:
        """Predict future production with enhanced parameter handling"""
        try:
            # Build payload with proper validation
            payload = {
                "well": well, 
                "economic_limit": float(economic_limit)
            }
            
            if selected_data:
                payload["selected_data"] = selected_data
                logger.info(f"Production prediction with selected data")
            
            logger.info(f"Sending prediction request: well={well}, ELR={economic_limit}")
            
            async with self.session.post(
                f"{self.base_url}/predict_production", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Prediction failed with status {response.status}: {error_text}")
                    return {"error": f"API returned status {response.status}: {error_text}"}
                
                data = await response.json()
                
                # Validate response
                if not isinstance(data, dict):
                    logger.error(f"Invalid prediction response type: {type(data)}")
                    return {"error": "Invalid response format"}
                
                if 'error' in data:
                    logger.error(f"Prediction API returned error: {data['error']}")
                    return data
                
                # Check for prediction data
                prediction_fields = ['ExponentialPrediction', 'HarmonicPrediction', 'HyperbolicPrediction']
                available_predictions = [field for field in prediction_fields if field in data and data[field]]
                
                if not available_predictions:
                    logger.warning("No prediction data returned")
                    return {"error": "No prediction data available"}
                
                logger.info(f"Prediction successful, available predictions: {available_predictions}")
                return data
                
        except Exception as e:
            logger.error(f"Error predicting production: {e}")
            return {"error": str(e)}
    
    async def predict_ml(self, elr: float = 10) -> Dict:
        """Get ML-based predictions with enhanced error handling"""
        try:
            payload = {"elr": float(elr)}
            
            logger.info(f"Sending ML prediction request with ELR: {elr}")
            
            async with self.session.post(
                f"{self.base_url}/predict_ml", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ML prediction failed with status {response.status}: {error_text}")
                    return {"error": f"API returned status {response.status}: {error_text}"}
                
                data = await response.json()
                
                # Validate response
                if not isinstance(data, dict):
                    logger.error(f"Invalid ML response type: {type(data)}")
                    return {"error": "Invalid response format"}
                
                if 'error' in data:
                    logger.error(f"ML API returned error: {data['error']}")
                    return data
                
                # Check for ML prediction data
                required_ml_fields = ['actual', 'predicted']
                missing_ml_fields = [field for field in required_ml_fields if field not in data]
                
                if missing_ml_fields:
                    logger.warning(f"ML response missing fields: {missing_ml_fields}")
                
                logger.info(f"ML prediction successful, response keys: {list(data.keys())}")
                return data
                
        except Exception as e:
            logger.error(f"Error getting ML predictions: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict:
        """Perform health check on the API"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("API health check passed")
                    return {"status": "healthy", "data": data}
                else:
                    logger.warning(f"API health check failed with status {response.status}")
                    return {"status": "unhealthy", "code": response.status}
                    
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {"status": "error", "error": str(e)}