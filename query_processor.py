import re
from typing import Dict, Any
from langdetect import detect
import logging

logger = logging.getLogger(__name__)

class DCAQueryProcessor:
    def __init__(self):
        self.intent_patterns = {
            "get_history": [
                "data sumur", "history", "historis", "production data", "data produksi",
                "show data", "tampilkan data", "produksi sumur", "well data",
                "historical production", "riwayat produksi"
            ],
            "analyze_decline": [
                "decline rate", "analisis dca", "dca analysis", "penurunan produksi",
                "decline curve", "automatic dca", "kurva penurunan", "analisa decline",
                "decline analysis", "model dca"
            ],
            "predict_future": [
                "estimasi", "predict", "prediksi", "forecast", "future production",
                "prediksi produksi", "production forecast", "ramalan produksi",
                "perkiraan", "proyeksi", "projection", "economic limit"
            ],
            "predict_ml": [
                "machine learning", "ml prediction", "prediksi ml", "neural network",
                "ai prediction", "artificial intelligence", "deep learning",
                "prediksi kecerdasan buatan", "model ml"
            ],
            "compare_wells": [
                "perbandingan", "compare", "terbaik", "best performing", "which well",
                "bandingkan sumur", "well comparison", "sumur terbaik", "ranking sumur",
                "performance comparison", "compare wells"
            ],
            "economic_analysis": [
                "economic limit", "batas ekonomis", "waktu ekonomis", "elr",
                "analisis ekonomi", "economic analysis", "revenue", "pendapatan"
            ]
        }
    
    def detect_language(self, query: str) -> str:
        """Detect language of the query"""
        try:
            return detect(query)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "id"  # Default to Indonesian
    
    def detect_intent(self, query: str) -> str:
        """Detect the intent of the user query with improved accuracy"""
        query_lower = query.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query_lower:
                    # Give higher score for exact matches
                    if pattern == query_lower.strip():
                        score += 10
                    else:
                        score += len(pattern.split())  # Multi-word patterns get higher scores
            intent_scores[intent] = score
        
        # Return intent with highest score, or general_dca if no clear match
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        return "general_dca"
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from the query with enhanced pattern matching"""
        params = {}
        
        # Extract well code (format: PKU00001-01 or variations)
        well_patterns = [
            r'PKU\d{5}-\d{2}',  # Standard format
            r'PKU-?\d{5}-?\d{2}',  # With optional hyphens
            r'pku\d{5}-\d{2}',  # Lowercase
            r'well\s+PKU\d{5}-\d{2}',  # With "well" prefix
            r'sumur\s+PKU\d{5}-\d{2}'  # With "sumur" prefix
        ]
        
        for pattern in well_patterns:
            well_match = re.search(pattern, query, re.IGNORECASE)
            if well_match:
                params["well"] = well_match.group().replace("well ", "").replace("sumur ", "").upper()
                break
        
        # Extract dates with multiple formats
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            found_dates = re.findall(pattern, query)
            dates.extend(found_dates)
        
        if len(dates) >= 2:
            params["start_date"] = self._normalize_date(dates[0])
            params["end_date"] = self._normalize_date(dates[1])
        elif len(dates) == 1:
            params["date"] = self._normalize_date(dates[0])
        
        # Extract time periods with multiple languages
        period_patterns = {
            "period_months": [
                (r'(\d+)\s*bulan', 1),
                (r'(\d+)\s*months?', 1),
                (r'(\d+)\s*tahun', 12),
                (r'(\d+)\s*years?', 12),
                (r'(\d+)\s*minggu', 0.25),
                (r'(\d+)\s*weeks?', 0.25),
                (r'6\s*bulan|six\s*months?|semester', 6),
                (r'3\s*bulan|three\s*months?|quarter', 3),
                (r'1\s*tahun|one\s*year|setahun', 12),
                (r'2\s*tahun|two\s*years?', 24)
            ]
        }
        
        for param_name, patterns in period_patterns.items():
            for pattern, multiplier in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    if match.groups():
                        params[param_name] = int(match.group(1)) * multiplier
                    else:
                        params[param_name] = multiplier
                    break
        
        # Extract economic limit with multiple patterns
        elr_patterns = [
            r'(?:economic\s*limit|elr|batas\s*ekonomis)[\s:]*(\d+(?:\.\d+)?)',
            r'(?:limit|batas)[\s:]*(\d+(?:\.\d+)?)\s*(?:bopd|barrel)',
            r'(\d+(?:\.\d+)?)\s*(?:bopd|barrel).*(?:limit|batas)'
        ]
        
        for pattern in elr_patterns:
            elr_match = re.search(pattern, query, re.IGNORECASE)
            if elr_match:
                params["economic_limit"] = float(elr_match.group(1))
                break
        
        # Extract ML specific parameters
        if "ml" in query.lower() or "machine learning" in query.lower():
            # Default ELR for ML if not specified
            if "economic_limit" not in params:
                params["economic_limit"] = 10.0
        
        # Extract comparison parameters
        if any(word in query.lower() for word in ["compare", "perbandingan", "terbaik", "best"]):
            # Look for number of wells to compare
            num_match = re.search(r'(?:top|teratas)\s*(\d+)', query, re.IGNORECASE)
            if num_match:
                params["top_wells"] = int(num_match.group(1))
            else:
                params["top_wells"] = 5  # default
        
        return params
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD format"""
        # Remove any extra spaces
        date_str = date_str.strip()
        
        # If already in YYYY-MM-DD format
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return date_str
        
        # Convert DD/MM/YYYY to YYYY-MM-DD
        if re.match(r'\d{2}/\d{2}/\d{4}', date_str):
            day, month, year = date_str.split('/')
            return f"{year}-{month}-{day}"
        
        # Convert DD-MM-YYYY to YYYY-MM-DD
        if re.match(r'\d{2}-\d{2}-\d{4}', date_str):
            day, month, year = date_str.split('-')
            return f"{year}-{month}-{day}"
        
        return date_str
    
    def is_dca_related(self, query: str) -> bool:
        """Check if query is related to DCA with improved detection"""
        dca_keywords = [
            # Core DCA terms
            "dca", "decline", "produksi", "production", "sumur", "well",
            "oil", "minyak", "bopd", "economic", "ekonomis", "predict",
            "prediksi", "forecast", "analisis", "analysis", "pku",
            
            # Technical terms
            "exponential", "harmonic", "hyperbolic", "decline rate",
            "economic limit", "elr", "machine learning", "ml",
            
            # Indonesian terms
            "kurva penurunan", "laju penurunan", "cadangan", "reservoir",
            "intervensi", "workover", "stimulation", "perforating",
            
            # Operational terms
            "history", "historis", "data", "comparison", "perbandingan",
            "performance", "performa", "optimization", "optimasi"
        ]
        
        query_lower = query.lower()
        
        # Check for direct keyword matches
        keyword_score = sum(1 for keyword in dca_keywords if keyword in query_lower)
        
        # Check for well codes
        well_code_pattern = r'pku\d{5}-\d{2}'
        well_code_match = re.search(well_code_pattern, query_lower)
        
        # Consider it DCA-related if:
        # 1. Has 2+ DCA keywords, OR
        # 2. Has 1+ keyword AND well code, OR  
        # 3. Has well code and production-related terms
        if keyword_score >= 2:
            return True
        elif keyword_score >= 1 and well_code_match:
            return True
        elif well_code_match and any(term in query_lower for term in ["production", "produksi", "data"]):
            return True
        
        return False
    
    def get_non_dca_response(self, lang: str) -> str:
        """Get response for non-DCA queries"""
        responses = {
            "id": """Maaf, saya adalah asisten DCA (Decline Curve Analysis) yang khusus membantu dengan:

🎯 **LAYANAN YANG TERSEDIA:**
• Analisis data historis produksi sumur
• Perhitungan decline curve analysis (Exponential, Harmonic, Hyperbolic)
• Prediksi produksi masa depan hingga economic limit
• Machine learning prediction untuk pola produksi kompleks
• Perbandingan performa antar sumur
• Analisis ekonomi dan rekomendasi intervensi

💡 **CONTOH PERTANYAAN:**
• "Tampilkan data historis sumur PKU00001-01"
• "Analisis DCA untuk sumur PKU00002-01"
• "Prediksi produksi sumur PKU00003-01 dengan economic limit 5 BOPD"
• "Bandingkan performa semua sumur dalam 6 bulan terakhir"
• "Prediksi ML untuk produksi dengan ELR 10 BOPD"

Silakan ajukan pertanyaan seputar DCA dan analisis produksi sumur.""",
            
            "en": """Sorry, I am a DCA (Decline Curve Analysis) assistant specialized in helping with:

🎯 **AVAILABLE SERVICES:**
• Well production historical data analysis
• Decline curve analysis calculations (Exponential, Harmonic, Hyperbolic)  
• Future production prediction to economic limit
• Machine learning prediction for complex production patterns
• Well performance comparison
• Economic analysis and intervention recommendations

💡 **EXAMPLE QUESTIONS:**
• "Show historical data for well PKU00001-01"
• "DCA analysis for well PKU00002-01"
• "Predict production for well PKU00003-01 with economic limit 5 BOPD"
• "Compare performance of all wells in the last 6 months"
• "ML prediction for production with ELR 10 BOPD"

Please ask questions about DCA and well production analysis."""
        }
        return responses.get(lang, responses["en"])
    
    def get_error_message(self, lang: str, error: str) -> str:
        """Get error message in appropriate language"""
        messages = {
            "id": f"Maaf, terjadi kesalahan saat memproses permintaan DCA Anda: {error}\n\nSilakan coba lagi atau hubungi administrator jika masalah berlanjut.",
            "en": f"Sorry, an error occurred while processing your DCA request: {error}\n\nPlease try again or contact the administrator if the problem persists."
        }
        return messages.get(lang, messages["en"])