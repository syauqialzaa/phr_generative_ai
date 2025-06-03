from urllib.parse import urlencode
from typing import Dict, Any

class WellboreUrlGenerator:
    def __init__(self):
        self.base_url = "https://syauqialzaa.github.io/wellbore"
    
    def generate_url(self, endpoint_type: str, params: Dict[str, Any]) -> str:
        """Generate URL for Wellbore application based on endpoint type and parameters"""
        
        if endpoint_type == "diagram":
            return self._generate_diagram_url(params)
        elif endpoint_type == "components":
            return self._generate_components_url(params)
        elif endpoint_type == "analysis":
            return self._generate_analysis_url(params)
        elif endpoint_type == "casing":
            return self._generate_casing_url(params)
        elif endpoint_type == "completion":
            return self._generate_completion_url(params)
        else:
            return self.base_url
    
    def _generate_diagram_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for wellbore diagram view"""
        url_params = {}
        
        if params.get("uwi"):
            url_params["uwi"] = params["uwi"]
        if params.get("top_md") is not None:
            url_params["top_md"] = str(params["top_md"])
        if params.get("bot_md") is not None:
            url_params["bot_md"] = str(params["bot_md"])
        
        # Add view parameter to specify diagram view
        url_params["view"] = "diagram"
        url_params["auto_load"] = "true"  # Automatically load diagram
        
        if url_params:
            return f"{self.base_url}/?{urlencode(url_params)}"
        return self.base_url
    
    def _generate_components_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for components analysis view"""
        url_params = {}
        
        if params.get("uwi"):
            url_params["uwi"] = params["uwi"]
        if params.get("component_filter"):
            url_params["component"] = params["component_filter"]
        if params.get("icon_name"):
            url_params["icon_filter"] = params["icon_name"]
        
        url_params["view"] = "components"
        url_params["auto_analyze"] = "true"
        
        if url_params:
            return f"{self.base_url}/?{urlencode(url_params)}"
        return self.base_url
    
    def _generate_analysis_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for wellbore analysis view"""
        url_params = {}
        
        if params.get("uwi"):
            url_params["uwi"] = params["uwi"]
        if params.get("analysis_type"):
            url_params["analysis"] = params["analysis_type"]
        
        url_params["view"] = "analysis"
        url_params["auto_run"] = "true"
        
        if url_params:
            return f"{self.base_url}/?{urlencode(url_params)}"
        return self.base_url
    
    def _generate_casing_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for casing analysis view"""
        url_params = {}
        
        if params.get("uwi"):
            url_params["uwi"] = params["uwi"]
        
        url_params["view"] = "casing"
        url_params["component"] = "casing"
        url_params["auto_analyze"] = "true"
        
        if url_params:
            return f"{self.base_url}/?{urlencode(url_params)}"
        return self.base_url
    
    def _generate_completion_url(self, params: Dict[str, Any]) -> str:
        """Generate URL for completion analysis view"""
        url_params = {}
        
        if params.get("uwi"):
            url_params["uwi"] = params["uwi"]
        
        url_params["view"] = "completion"
        url_params["auto_analyze"] = "true"
        
        if url_params:
            return f"{self.base_url}/?{urlencode(url_params)}"
        return self.base_url
    
    def generate_api_url(self, endpoint: str, params: Dict[str, Any] = None) -> str:
        """Generate direct API URL for testing or external access"""
        api_base = "https://857949f11264.ngrok.app"
        
        if endpoint == "wellbore_data":
            url = f"{api_base}/api/wellbore-data"
            if params:
                return f"{url}?{urlencode(params)}"
            return url
        elif endpoint == "icons":
            url = f"{api_base}/api/icons"
            if params:
                return f"{url}?{urlencode(params)}"
            return url
        elif endpoint == "icon_image":
            filename = params.get("filename", "") if params else ""
            return f"{api_base}/img/{filename}"
        else:
            return api_base
    
    def generate_documentation_url(self) -> str:
        """Generate URL for wellbore documentation"""
        return f"{self.base_url}/docs"
    
    def generate_health_check_url(self) -> str:
        """Generate URL for health check"""
        return "https://857949f11264.ngrok.app/health"