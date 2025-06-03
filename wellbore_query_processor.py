import re
from typing import Dict, Any
from langdetect import detect
import logging

logger = logging.getLogger(__name__)

class WellboreQueryProcessor:
    def __init__(self):
        self.wellbore_patterns = {
            "show_diagram": [
                "wellbore diagram", "diagram sumur", "skema sumur", "wellbore schematic",
                "tampilkan diagram", "show diagram", "gambar sumur", "well diagram",
                "wellbore visualization", "visualisasi sumur", "struktur sumur", "well structure"
            ],
            "wellbore_analysis": [
                "analisis wellbore", "wellbore analysis", "analisa sumur", "well analysis",
                "komponen sumur", "well components", "konfigurasi sumur", "well configuration",
                "wellbore components", "struktur wellbore"
            ],
            "casing_analysis": [
                "casing", "selubung", "casing design", "desain casing", "casing size",
                "ukuran casing", "casing depth", "kedalaman casing", "casing string"
            ],
            "completion_analysis": [
                "completion", "komplesi", "perforations", "perforasi", "tubing",
                "artificial lift", "esp", "pump", "pompa", "motor"
            ],
            "depth_analysis": [
                "depth", "kedalaman", "md", "tvd", "total depth", "kedalaman total",
                "measured depth", "true vertical depth"
            ]
        }
    
    def detect_language(self, query: str) -> str:
        """Detect language of the query"""
        try:
            return detect(query)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "id"  # Default to Indonesian
    
    def is_wellbore_related(self, query: str) -> bool:
        """Check if query is related to wellbore diagrams or analysis"""
        wellbore_keywords = [
            # Core wellbore terms
            "wellbore", "sumur", "well", "diagram", "skema", "casing", "selubung",
            "tubing", "completion", "komplesi", "perforations", "perforasi",
            "esp", "pump", "pompa", "motor", "packer", "seal",
            
            # Technical terms
            "surface casing", "intermediate casing", "production casing",
            "artificial lift", "downhole", "pbtd", "perforation", "intake",
            
            # Indonesian terms
            "komponen sumur", "struktur sumur", "konfigurasi sumur",
            "analisis sumur", "visualisasi sumur", "gambaran sumur",
            
            # UWI patterns
            "peb", "uwi"
        ]
        
        query_lower = query.lower()
        
        # Check for direct keyword matches
        keyword_score = sum(1 for keyword in wellbore_keywords if keyword in query_lower)
        
        # Check for UWI patterns (PEB000026D1, etc.)
        uwi_pattern = r'peb\d{6}[a-z]\d'
        uwi_match = re.search(uwi_pattern, query_lower)
        
        # Consider it wellbore-related if:
        # 1. Has 2+ wellbore keywords, OR
        # 2. Has 1+ keyword AND UWI, OR
        # 3. Has UWI and diagram/visual terms
        if keyword_score >= 2:
            return True
        elif keyword_score >= 1 and uwi_match:
            return True
        elif uwi_match and any(term in query_lower for term in ["diagram", "visual", "show", "tampil"]):
            return True
        
        return False
    
    def detect_wellbore_intent(self, query: str) -> str:
        """Detect the intent of wellbore-related queries"""
        query_lower = query.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        for intent, patterns in self.wellbore_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query_lower:
                    # Give higher score for exact matches
                    if pattern == query_lower.strip():
                        score += 10
                    else:
                        score += len(pattern.split())
            intent_scores[intent] = score
        
        # Return intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        return "show_diagram"  # Default to showing diagram
    
    def extract_wellbore_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from wellbore queries"""
        params = {}
        
        # Extract UWI (Universal Well Identifier)
        uwi_patterns = [
            r'PEB\d{6}[A-Z]\d',  # Standard format (PEB000026D1)
            r'peb\d{6}[a-z]\d',  # Lowercase
            r'PEB-?\d{6}-?[A-Z]\d',  # With optional hyphens
            r'uwi\s+PEB\d{6}[A-Z]\d',  # With "UWI" prefix
            r'sumur\s+PEB\d{6}[A-Z]\d'  # With "sumur" prefix
        ]
        
        for pattern in uwi_patterns:
            uwi_match = re.search(pattern, query, re.IGNORECASE)
            if uwi_match:
                params["uwi"] = uwi_match.group().replace("uwi ", "").replace("sumur ", "").upper()
                break
        
        # Extract depth parameters
        depth_patterns = [
            (r'(?:dari|from)\s*(\d+)\s*(?:hingga|to|sampai)\s*(\d+)\s*(?:ft|feet|kaki)?', 'range'),
            (r'(?:depth|kedalaman)\s*(\d+)\s*(?:ft|feet|kaki)?', 'single'),
            (r'(?:top|atas)\s*(\d+)\s*(?:ft|feet|kaki)?', 'top'),
            (r'(?:bottom|bawah|bot)\s*(\d+)\s*(?:ft|feet|kaki)?', 'bottom'),
            (r'(\d+)\s*-\s*(\d+)\s*(?:ft|feet|kaki)?', 'range')
        ]
        
        for pattern, depth_type in depth_patterns:
            depth_match = re.search(pattern, query, re.IGNORECASE)
            if depth_match:
                if depth_type == 'range':
                    params["top_md"] = int(depth_match.group(1))
                    params["bot_md"] = int(depth_match.group(2))
                elif depth_type == 'top':
                    params["top_md"] = int(depth_match.group(1))
                elif depth_type == 'bottom':
                    params["bot_md"] = int(depth_match.group(1))
                elif depth_type == 'single':
                    params["target_depth"] = int(depth_match.group(1))
                break
        
        # Extract component filters
        component_patterns = {
            "casing": ["casing", "selubung", "surface casing", "production casing", "intermediate casing"],
            "tubing": ["tubing"],
            "esp": ["esp", "pump", "pompa", "motor"],
            "perforation": ["perforation", "perforasi", "perfo"],
            "packer": ["packer", "pkr"],
            "seal": ["seal"]
        }
        
        for comp_type, keywords in component_patterns.items():
            for keyword in keywords:
                if keyword in query.lower():
                    params["component_filter"] = comp_type
                    params["icon_name"] = keyword
                    break
            if "component_filter" in params:
                break
        
        # Extract analysis type
        analysis_patterns = {
            "summary": ["summary", "ringkasan", "overview", "gambaran"],
            "detailed": ["detail", "detailed", "lengkap", "comprehensive", "komprehensif"],
            "components": ["components", "komponen", "parts", "bagian"],
            "configuration": ["configuration", "konfigurasi", "setup", "arrangement"]
        }
        
        for analysis_type, keywords in analysis_patterns.items():
            for keyword in keywords:
                if keyword in query.lower():
                    params["analysis_type"] = analysis_type
                    break
            if "analysis_type" in params:
                break
        
        # Set default UWI if not specified
        if "uwi" not in params:
            params["uwi"] = "PEB000026D1"  # Default UWI
        
        return params
    
    def get_non_wellbore_response(self, lang: str) -> str:
        """Get response for non-wellbore queries"""
        responses = {
            "id": """Maaf, saya adalah asisten yang dapat membantu dengan analisis DCA dan Wellbore Diagram. 

🎯 **LAYANAN WELLBORE YANG TERSEDIA:**
• Visualisasi diagram wellbore dengan komponen lengkap
• Analisis konfigurasi sumur dan komponen
• Analisis casing, tubing, dan completion
• Identifikasi artificial lift systems (ESP)
• Evaluasi struktur wellbore dan rekomendasi

💡 **CONTOH PERTANYAAN WELLBORE:**
• "Tampilkan diagram wellbore untuk sumur PEB000026D1"
• "Analisis komponen wellbore dari kedalaman 0-5000 ft"
• "Show wellbore diagram dengan komponen casing"
• "Konfigurasi completion untuk sumur PEB000026D1"
• "Wellbore analysis untuk ESP components"

🔧 **UNTUK ANALISIS DCA:**
• Decline curve analysis dan prediksi produksi
• Machine learning predictions
• Perbandingan performa sumur

Silakan ajukan pertanyaan seputar wellbore diagram atau DCA analysis.""",
            
            "en": """Sorry, I am an assistant that can help with DCA analysis and Wellbore Diagrams.

🎯 **AVAILABLE WELLBORE SERVICES:**
• Wellbore diagram visualization with complete components
• Well configuration and component analysis  
• Casing, tubing, and completion analysis
• Artificial lift systems (ESP) identification
• Wellbore structure evaluation and recommendations

💡 **EXAMPLE WELLBORE QUESTIONS:**
• "Show wellbore diagram for well PEB000026D1"
• "Analyze wellbore components from depth 0-5000 ft"
• "Wellbore diagram with casing components"
• "Completion configuration for well PEB000026D1"
• "Wellbore analysis for ESP components"

🔧 **FOR DCA ANALYSIS:**
• Decline curve analysis and production predictions
• Machine learning predictions
• Well performance comparisons

Please ask questions about wellbore diagrams or DCA analysis."""
        }
        return responses.get(lang, responses["en"])
    
    def get_error_message(self, lang: str, error: str) -> str:
        """Get error message in appropriate language"""
        messages = {
            "id": f"Maaf, terjadi kesalahan saat memproses permintaan wellbore Anda: {error}\n\nSilakan coba lagi atau hubungi administrator jika masalah berlanjut.",
            "en": f"Sorry, an error occurred while processing your wellbore request: {error}\n\nPlease try again or contact the administrator if the problem persists."
        }
        return messages.get(lang, messages["en"])