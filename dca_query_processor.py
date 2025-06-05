import re
from typing import Dict, Any
from langdetect import detect
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DCAQueryProcessor:
    def __init__(self):
        self.intent_patterns = {
            "get_history": [
                "data sumur", "history", "historis", "production data", "data produksi",
                "show data", "tampilkan data", "produksi sumur", "well data",
                "historical production", "riwayat produksi", "grafik", "chart"
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
        """Extract parameters from the query with ENHANCED date processing"""
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
        
        # ===== ENHANCED DATE EXTRACTION =====
        # First, try to extract date ranges with various separators and formats
        date_range_patterns = [
            # Pattern: dari/from X sampai/to Y
            r'(?:dari|from)\s+([^\s]+(?:\s+\d{4})?)\s+(?:sampai|hingga|to)\s+([^\s]+(?:\s+\d{4})?)',
            # Pattern: tanggal X - Y or X sampai Y
            r'(?:tanggal\s+)?([^\s]+(?:\s+\d{4})?)\s+(?:sampai|hingga|-|to)\s+([^\s]+(?:\s+\d{4})?)',
            # Pattern: periode X to Y
            r'(?:periode\s+)?([^\s]+(?:\s+\d{4})?)\s+(?:sampai|hingga|to|-)\s+([^\s]+(?:\s+\d{4})?)'
        ]
        
        dates_found = False
        for pattern in date_range_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                start_date_str = match.group(1).strip()
                end_date_str = match.group(2).strip()
                
                # Parse and normalize dates
                start_date = self._parse_flexible_date(start_date_str)
                end_date = self._parse_flexible_date(end_date_str)
                
                if start_date and end_date:
                    params["start_date"] = start_date
                    params["end_date"] = end_date
                    dates_found = True
                    logger.info(f"Extracted date range: {start_date} to {end_date}")
                    break
        
        # If no date range found, try individual date patterns
        if not dates_found:
            date_patterns = [
                r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD or YYYY-M-D
                r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # DD/MM/YYYY or DD-MM-YYYY
                r'\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}',  # Indonesian months
                r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',  # English months
                r'(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}',  # Month Year (Indonesian)
                r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'  # Month Year (English)
            ]
            
            all_dates = []
            for pattern in date_patterns:
                found_dates = re.findall(pattern, query, re.IGNORECASE)
                all_dates.extend(found_dates)
            
            # Remove duplicates while preserving order
            unique_dates = []
            for date in all_dates:
                if date not in unique_dates:
                    unique_dates.append(date)
            
            if len(unique_dates) >= 2:
                start_date = self._parse_flexible_date(unique_dates[0])
                end_date = self._parse_flexible_date(unique_dates[1])
                if start_date and end_date:
                    params["start_date"] = start_date
                    params["end_date"] = end_date
                    logger.info(f"Extracted individual dates: {start_date} to {end_date}")
            elif len(unique_dates) == 1:
                parsed_date = self._parse_flexible_date(unique_dates[0])
                if parsed_date:
                    params["date"] = parsed_date
        
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
        
        logger.info(f"Final extracted parameters: {params}")
        return params
    
    def _parse_flexible_date(self, date_str: str) -> str:
        """Parse various date formats and return YYYY-MM-DD format"""
        date_str = date_str.strip()
        
        # Indonesian month mapping
        indonesian_months = {
            'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
            'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
            'september': '09', 'oktober': '10', 'november': '11', 'desember': '12'
        }
        
        # English month mapping
        english_months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        try:
            # Case 1: Already in YYYY-MM-DD format
            if re.match(r'\d{4}-\d{1,2}-\d{1,2}', date_str):
                parts = date_str.split('-')
                year, month, day = parts[0], parts[1].zfill(2), parts[2].zfill(2)
                return f"{year}-{month}-{day}"
            
            # Case 2: DD/MM/YYYY or DD-MM-YYYY format
            if re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', date_str):
                separator = '/' if '/' in date_str else '-'
                parts = date_str.split(separator)
                day, month, year = parts[0].zfill(2), parts[1].zfill(2), parts[2]
                return f"{year}-{month}-{day}"
            
            # Case 3: DD Month YYYY (Indonesian)
            pattern = r'(\d{1,2})\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+(\d{4})'
            match = re.match(pattern, date_str.lower())
            if match:
                day = match.group(1).zfill(2)
                month = indonesian_months[match.group(2)]
                year = match.group(3)
                return f"{year}-{month}-{day}"
            
            # Case 4: DD Month YYYY (English)
            pattern = r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
            match = re.match(pattern, date_str.lower())
            if match:
                day = match.group(1).zfill(2)
                month = english_months[match.group(2)]
                year = match.group(3)
                return f"{year}-{month}-{day}"
            
            # Case 5: Month YYYY (Indonesian) - assume first day of month
            pattern = r'(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+(\d{4})'
            match = re.match(pattern, date_str.lower())
            if match:
                month = indonesian_months[match.group(1)]
                year = match.group(2)
                return f"{year}-{month}-01"
            
            # Case 6: Month YYYY (English) - assume first day of month
            pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
            match = re.match(pattern, date_str.lower())
            if match:
                month = english_months[match.group(1)]
                year = match.group(2)
                return f"{year}-{month}-01"
            
            # Case 7: Special format D-M-YYYY
            if re.match(r'\d{1,2}-\d{1,2}-\d{4}', date_str):
                parts = date_str.split('-')
                day, month, year = parts[0].zfill(2), parts[1].zfill(2), parts[2]
                return f"{year}-{month}-{day}"
            
            logger.warning(f"Could not parse date: {date_str}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing date '{date_str}': {e}")
            return None
    
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
            "performance", "performa", "optimization", "optimasi",
            "grafik", "chart", "riwayat"
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
        elif well_code_match and any(term in query_lower for term in ["production", "produksi", "data", "grafik", "chart"]):
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