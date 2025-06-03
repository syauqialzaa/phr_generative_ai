from datetime import datetime
from typing import Dict, List
import logging

from dca_assistant import DCAAssistant
from wellbore_assistant import WellboreAssistant
from dca_query_processor import DCAQueryProcessor
from wellbore_query_processor import WellboreQueryProcessor

logger = logging.getLogger(__name__)

class UnifiedAssistant:
    """Unified assistant that handles both DCA and Wellbore queries"""
    
    def __init__(self):
        self.dca_assistant = DCAAssistant()
        self.wellbore_assistant = WellboreAssistant()
        self.dca_query_processor = DCAQueryProcessor()
        self.wellbore_query_processor = WellboreQueryProcessor()
    
    async def process_query(self, query: str, history: List[Dict] = None) -> Dict:
        """Process query by routing to appropriate assistant"""
        
        try:
            # Detect language first
            detected_lang = self.dca_query_processor.detect_language(query)
            
            # Check if query is wellbore-related first (more specific)
            if self.wellbore_query_processor.is_wellbore_related(query):
                logger.info(f"Routing query to Wellbore Assistant: {query}")
                return await self.wellbore_assistant.process_query(query, history)
            
            # Check if query is DCA-related
            elif self.dca_query_processor.is_dca_related(query):
                logger.info(f"Routing query to DCA Assistant: {query}")
                return await self.dca_assistant.process_query(query, history)
            
            # Neither DCA nor wellbore related
            else:
                logger.info(f"Query not related to DCA or Wellbore: {query}")
                return {
                    "type": "response",
                    "explanation": self._get_general_non_related_response(detected_lang),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in unified query processing: {e}")
            return {
                "type": "error",
                "message": f"Error processing query: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_general_non_related_response(self, lang: str) -> str:
        """Get response for queries that are neither DCA nor wellbore related"""
        responses = {
            "id": """Maaf, saya adalah asisten khusus untuk analisis DCA (Decline Curve Analysis) dan Wellbore Diagram.

🎯 **LAYANAN DCA YANG TERSEDIA:**
• Analisis data historis produksi sumur
• Perhitungan decline curve analysis (Exponential, Harmonic, Hyperbolic)
• Prediksi produksi masa depan hingga economic limit
• Machine learning prediction untuk pola produksi kompleks
• Perbandingan performa antar sumur

🏗️ **LAYANAN WELLBORE YANG TERSEDIA:**
• Visualisasi diagram wellbore dengan komponen lengkap
• Analisis konfigurasi sumur dan komponen
• Analisis casing, tubing, dan completion
• Identifikasi artificial lift systems (ESP)
• Evaluasi struktur wellbore dan rekomendasi

💡 **CONTOH PERTANYAAN:**

**DCA Analysis:**
• "Tampilkan data historis sumur PKU00001-01"
• "Analisis DCA untuk sumur PKU00002-01"
• "Prediksi produksi sumur PKU00003-01 dengan economic limit 5 BOPD"
• "ML prediction untuk produksi dengan ELR 10 BOPD"
• "Bandingkan performa semua sumur dalam 6 bulan terakhir"

**Wellbore Analysis:**
• "Tampilkan diagram wellbore untuk sumur PEB000026D1"
• "Analisis komponen wellbore dari kedalaman 0-5000 ft"
• "Show wellbore diagram dengan komponen casing"
• "Konfigurasi completion untuk sumur PEB000026D1"
• "Wellbore analysis untuk ESP components"

Silakan ajukan pertanyaan seputar DCA analysis atau wellbore diagrams.""",
            
            "en": """Sorry, I am a specialized assistant for DCA (Decline Curve Analysis) and Wellbore Diagrams.

🎯 **AVAILABLE DCA SERVICES:**
• Well production historical data analysis
• Decline curve analysis calculations (Exponential, Harmonic, Hyperbolic)
• Future production prediction to economic limit
• Machine learning prediction for complex production patterns
• Well performance comparisons

🏗️ **AVAILABLE WELLBORE SERVICES:**
• Wellbore diagram visualization with complete components
• Well configuration and component analysis
• Casing, tubing, and completion analysis
• Artificial lift systems (ESP) identification
• Wellbore structure evaluation and recommendations

💡 **EXAMPLE QUESTIONS:**

**DCA Analysis:**
• "Show historical data for well PKU00001-01"
• "DCA analysis for well PKU00002-01"
• "Predict production for well PKU00003-01 with economic limit 5 BOPD"
• "ML prediction for production with ELR 10 BOPD"
• "Compare performance of all wells in the last 6 months"

**Wellbore Analysis:**
• "Show wellbore diagram for well PEB000026D1"
• "Analyze wellbore components from depth 0-5000 ft"
• "Wellbore diagram with casing components"
• "Completion configuration for well PEB000026D1"
• "Wellbore analysis for ESP components"

Please ask questions about DCA analysis or wellbore diagrams."""
        }
        return responses.get(lang, responses["en"])