import aiohttp
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class WellboreApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_wellbore_data(self, uwi: str = None, top_md: int = None, 
                               bot_md: int = None, icon_name: str = None) -> List[Dict]:
        """Get wellbore component data with filtering"""
        try:
            params = {}
            if uwi:
                params["uwi"] = uwi
            if top_md is not None:
                params["top_md"] = top_md
            if bot_md is not None:
                params["bot_md"] = bot_md
            if icon_name:
                params["icon_name"] = icon_name
            
            async with self.session.get(f"{self.base_url}/api/wellbore-data", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error getting wellbore data: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting wellbore data: {e}")
            return []
    
    async def get_icons(self, search: str = None) -> List[str]:
        """Get list of available icon files"""
        try:
            params = {}
            if search:
                params["search"] = search
            
            async with self.session.get(f"{self.base_url}/api/icons", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error getting icons: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting icons: {e}")
            return []
    
    async def get_icon_image(self, filename: str) -> bytes:
        """Get icon image data"""
        try:
            async with self.session.get(f"{self.base_url}/img/{filename}") as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Error getting icon image {filename}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting icon image {filename}: {e}")
            return None