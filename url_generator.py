from urllib.parse import urlencode
from typing import Dict, Any
from config import DCA_APP_BASE_URL

class DCAUrlGenerator:
    def __init__(self):
        self.base_url = DCA_APP_BASE_URL.rstrip('/')
    
    def generate_url(self, endpoint_type: str, params: Dict[str, Any]) -> str:
        """Generate URL for DCA application based on endpoint type and parameters"""
        
        if endpoint_type == "history":
            return self._generate_history_url(params)
        elif endpoint_type == "dca":
            return self._generate_dca_url(params)
        elif endpoint_type == "prediction":
            return self._generate_prediction_url(params)
        elif endpoint_type == "ml_prediction":
            return self._generate_ml_prediction_url(params)
        elif endpoint_type == "comparison":
            return self._generate_comparison_url(params)
        else:
            return self.base_url
    
    def _generate_history_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for production history view"""
        url_params = {}
        
        if params.get("well"):
            url_params["well"] = params["well"]
        if params.get("start_date"):
            url_params["start_date"] = params["start_date"]
        if params.get("end_date"):
            url_params["end_date"] = params["end_date"]
        
        # Add view parameter to specify history view
        url_params["view"] = "history"
        
        if url_params:
            return f"{self.base_url}/?{urlencode(url_params)}"
        return self.base_url
    
    def _generate_dca_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for DCA analysis view"""
        url_params = {}
        
        if params.get("well"):
            url_params["well"] = params["well"]
            url_params["view"] = "dca"
            url_params["auto_run"] = "true"  # Automatically run DCA analysis
        
        if params.get("start_date"):
            url_params["start_date"] = params["start_date"]
        if params.get("end_date"):
            url_params["end_date"] = params["end_date"]
        
        if url_params:
            return f"{self.base_url}/?{urlencode(url_params)}"
        return self.base_url
    
    def _generate_prediction_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for production prediction view"""
        url_params = {}
        
        if params.get("well"):
            url_params["well"] = params["well"]
            url_params["view"] = "prediction"
        
        if params.get("economic_limit") or params.get("elr"):
            elr = params.get("economic_limit") or params.get("elr")
            url_params["economic_limit"] = str(elr)
        
        # Add auto-run parameter for predictions
        if params.get("well"):
            url_params["auto_predict"] = "true"
        
        if url_params:
            return f"{self.base_url}/?{urlencode(url_params)}"
        return self.base_url
    
    def _generate_ml_prediction_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for ML prediction view"""
        url_params = {
            "view": "ml_prediction"
        }
        
        if params.get("economic_limit") or params.get("elr"):
            elr = params.get("economic_limit") or params.get("elr")
            url_params["elr"] = str(elr)
        else:
            url_params["elr"] = "10.0"  # Default ELR for ML
        
        # Auto-run ML prediction
        url_params["auto_run_ml"] = "true"
        
        return f"{self.base_url}/?{urlencode(url_params)}"
    
    def _generate_comparison_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for well comparison view"""
        url_params = {
            "view": "comparison"
        }
        
        if params.get("period_months"):
            url_params["period"] = str(params["period_months"])
        else:
            url_params["period"] = "6"  # Default 6 months
        
        if params.get("top_wells"):
            url_params["top_wells"] = str(params["top_wells"])
        
        # Auto-run comparison
        url_params["auto_compare"] = "true"
        
        return f"{self.base_url}/?{urlencode(url_params)}"
    
    def generate_api_url(self, endpoint: str, params: Dict[str, Any] = None) -> str:
        """Generate direct API URL for testing or external access"""
        api_base = self.base_url.replace('/app', '/api')  # Assuming API is at /api
        
        if endpoint == "get_wells":
            return f"{api_base}/get_wells"
        elif endpoint == "get_history":
            url = f"{api_base}/get_history"
            if params:
                return f"{url}?{urlencode(params)}"
            return url
        elif endpoint == "automatic_dca":
            url = f"{api_base}/automatic_dca"
            if params and params.get("well"):
                return f"{url}?{urlencode(params)}"
            return url
        elif endpoint == "predict_production":
            url = f"{api_base}/predict_production"
            if params:
                return f"{url}?{urlencode(params)}"
            return url
        elif endpoint == "predict_ml":
            url = f"{api_base}/predict_ml"
            if params:
                return f"{url}?{urlencode(params)}"
            return url
        else:
            return api_base
    
    def generate_documentation_url(self) -> str:
        """Generate URL for API documentation"""
        return f"{self.base_url}/docs"
    
    def generate_health_check_url(self) -> str:
        """Generate URL for health check"""
        return f"{self.base_url}/health"