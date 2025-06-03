from urllib.parse import urlencode
from typing import Dict
from config import DCA_APP_BASE_URL

class DCAUrlGenerator:
    @staticmethod
    def generate_url(view_type: str, params: Dict) -> str:
        """Generate interactive DCA app URL with parameters"""
        base_url = DCA_APP_BASE_URL
        
        # Map view types
        view_map = {
            "history": "history",
            "dca": "dca", 
            "prediction": "prediction",
            "ml": "ml"
        }
        
        url_params = {"view": view_map.get(view_type, "history")}
        
        # Add other parameters
        if "well" in params:
            url_params["well"] = params["well"]
        if "start_date" in params:
            url_params["start_date"] = params["start_date"]
        if "end_date" in params:
            url_params["end_date"] = params["end_date"]
        if "elr" in params:
            url_params["elr"] = params["elr"]
        if "economic_limit" in params:
            url_params["elr"] = params["economic_limit"]
        
        # Build URL
        query_string = urlencode(url_params)
        return f"{base_url}/?{query_string}"