import re
from typing import Dict, Any
from langdetect import detect
import logging

logger = logging.getLogger(__name__)

class DCAQueryProcessor:
    def __init__(self):
        self.intent_patterns = {
            "get_history": ["data sumur", "history", "historis", "production data", "data produksi"],
            "analyze_decline": ["decline rate", "analisis dca", "dca analysis", "penurunan produksi"],
            "predict_future": ["estimasi", "predict", "prediksi", "forecast", "future production"],
            "compare_wells": ["perbandingan", "compare", "terbaik", "best performing", "which well"],
            "economic_analysis": ["economic limit", "batas ekonomis", "waktu ekonomis", "elr"]
        }
    
    def detect_language(self, query: str) -> str:
        """Detect language of the query"""
        try:
            return detect(query)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "id"  # Default to Indonesian
    
    def detect_intent(self, query: str) -> str:
        """Detect the intent of the user query"""
        query_lower = query.lower()
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        return "general_dca"
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from the query"""
        params = {}
        
        # Extract well code (format: PKU00001-01)
        well_pattern = r'PKU\d{5}-\d{2}'
        well_match = re.search(well_pattern, query, re.IGNORECASE)
        if well_match:
            params["well"] = well_match.group()
        
        # Extract dates
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, query)
        if len(dates) >= 2:
            params["start_date"] = dates[0]
            params["end_date"] = dates[1]
        elif len(dates) == 1:
            params["date"] = dates[0]
        
        # Extract time periods
        if "6 bulan" in query or "six months" in query:
            params["period_months"] = 6
        elif "3 bulan" in query or "three months" in query:
            params["period_months"] = 3
        elif "1 tahun" in query or "one year" in query:
            params["period_months"] = 12
        
        # Extract economic limit
        elr_match = re.search(r'(?:economic limit|elr|batas ekonomis)[\s:]*(\d+)', query, re.IGNORECASE)
        if elr_match:
            params["economic_limit"] = float(elr_match.group(1))
        
        return params
    
    def is_dca_related(self, query: str) -> bool:
        """Check if query is related to DCA"""
        dca_keywords = [
            "dca", "decline", "produksi", "production", "sumur", "well",
            "oil", "minyak", "bopd", "economic", "ekonomis", "predict",
            "prediksi", "forecast", "analisis", "analysis", "pku"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in dca_keywords)
    
    def get_non_dca_response(self, lang: str) -> str:
        """Get response for non-DCA queries"""
        responses = {
            "id": "Maaf, saya hanya dapat membantu dengan pertanyaan seputar DCA (Decline Curve Analysis) dan analisis produksi sumur minyak. Silakan ajukan pertanyaan terkait DCA.",
            "en": "Sorry, I can only help with questions about DCA (Decline Curve Analysis) and oil well production analysis. Please ask DCA-related questions."
        }
        return responses.get(lang, responses["en"])
    
    def get_error_message(self, lang: str, error: str) -> str:
        """Get error message in appropriate language"""
        messages = {
            "id": f"Maaf, terjadi kesalahan saat memproses permintaan Anda: {error}",
            "en": f"Sorry, an error occurred while processing your request: {error}"
        }
        return messages.get(lang, messages["en"])