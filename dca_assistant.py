import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

from dca_client import DCAApiClient
from query_processor import DCAQueryProcessor
from visualization import VisualizationGenerator
from url_generator import DCAUrlGenerator
from vector_store import VectorStoreManager
from config import DCA_API_BASE_URL, DCA_APP_BASE_URL, MILVUS_DB_PATH

logger = logging.getLogger(__name__)

class DCAAssistant:
    def __init__(self):
        self.query_processor = DCAQueryProcessor()
        self.viz_generator = VisualizationGenerator()
        self.url_generator = DCAUrlGenerator()
        self.vector_store = VectorStoreManager(MILVUS_DB_PATH)
    
    async def process_query(self, query: str, history: List[Dict] = None) -> Dict:
        """Process DCA-related query and return comprehensive response"""
        
        # Detect language
        detected_lang = self.query_processor.detect_language(query)
        
        # Check if query is DCA-related
        if not self.query_processor.is_dca_related(query):
            return {
                "type": "response",
                "explanation": self.query_processor.get_non_dca_response(detected_lang),
                "timestamp": datetime.now().isoformat()
            }
        
        # Detect intent and extract parameters
        intent = self.query_processor.detect_intent(query)
        params = self.query_processor.extract_parameters(query)
        
        # Process based on intent
        try:
            async with DCAApiClient(DCA_API_BASE_URL) as client:
                if intent == "get_history":
                    return await self._handle_get_history(client, params, detected_lang)
                elif intent == "analyze_decline":
                    return await self._handle_analyze_decline(client, params, detected_lang)
                elif intent == "predict_future":
                    return await self._handle_predict_future(client, params, detected_lang)
                elif intent == "compare_wells":
                    return await self._handle_compare_wells(client, params, detected_lang)
                else:
                    return await self._handle_general_dca(query, detected_lang)
                    
        except Exception as e:
            logger.error(f"Error processing DCA query: {e}")
            return {
                "type": "error",
                "message": self.query_processor.get_error_message(detected_lang, str(e)),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _handle_get_history(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle production history queries"""
        # Get history data
        history_data = await client.get_history(
            well=params.get("well"),
            start_date=params.get("start_date"),
            end_date=params.get("end_date")
        )
        
        if not history_data:
            return {
                "type": "response",
                "explanation": "No data found for the specified parameters.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Create visualization
        chart_base64 = self.viz_generator.create_production_chart(
            history_data,
            f"Production History - {params.get('well', 'All Wells')}"
        )
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("history", params)
        
        # Create summary
        df = pd.DataFrame(history_data)
        avg_production = df['Production'].mean()
        max_production = df['Production'].max()
        min_production = df['Production'].min()
        
        if lang == "id":
            explanation = f"""Berdasarkan data historis sumur {params.get('well', '')} dari {params.get('start_date', df['Date'].min())} sampai {params.get('end_date', df['Date'].max())}:

• Rata-rata produksi: {avg_production:.2f} BOPD
• Produksi maksimum: {max_production:.2f} BOPD
• Produksi minimum: {min_production:.2f} BOPD
• Total data points: {len(df)}

Grafik menunjukkan tren produksi minyak (hijau) dan fluida (biru) sepanjang periode tersebut."""
        else:
            explanation = f"""Based on historical data for well {params.get('well', '')} from {params.get('start_date', df['Date'].min())} to {params.get('end_date', df['Date'].max())}:

• Average production: {avg_production:.2f} BOPD
• Maximum production: {max_production:.2f} BOPD
• Minimum production: {min_production:.2f} BOPD
• Total data points: {len(df)}

The chart shows oil (green) and fluid (blue) production trends over the period."""
        
        return {
            "type": "response",
            "explanation": explanation,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_analyze_decline(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle DCA analysis queries"""
        if not params.get("well"):
            return {
                "type": "response",
                "explanation": "Please specify a well code for DCA analysis.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get DCA analysis
        dca_result = await client.automatic_dca(params["well"])
        
        # Extract decline rates
        decline_rates = dca_result.get("DeclineRate", {})
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("dca", params)
        
        if lang == "id":
            explanation = f"""Hasil analisis DCA untuk sumur {params['well']}:

• Decline Rate Exponential: {decline_rates.get('Exponential', 'N/A')}%/tahun
• Decline Rate Harmonic: {decline_rates.get('Harmonic', 'N/A')}%/tahun
• Decline Rate Hyperbolic: {decline_rates.get('Hyperbolic', 'N/A')}%/tahun

Model Exponential umumnya memberikan estimasi konservatif, sementara Harmonic dan Hyperbolic dapat memberikan proyeksi yang lebih optimis. Pemilihan model tergantung pada karakteristik reservoir dan sejarah produksi sumur."""
        else:
            explanation = f"""DCA analysis results for well {params['well']}:

• Exponential Decline Rate: {decline_rates.get('Exponential', 'N/A')}%/year
• Harmonic Decline Rate: {decline_rates.get('Harmonic', 'N/A')}%/year
• Hyperbolic Decline Rate: {decline_rates.get('Hyperbolic', 'N/A')}%/year

The Exponential model typically provides conservative estimates, while Harmonic and Hyperbolic models may give more optimistic projections. Model selection depends on reservoir characteristics and production history."""
        
        return {
            "type": "response",
            "explanation": explanation,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_predict_future(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle production prediction queries"""
        if not params.get("well"):
            return {
                "type": "response",
                "explanation": "Please specify a well code for prediction.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get predictions
        economic_limit = params.get("economic_limit", 5)
        prediction_result = await client.predict_production(
            params["well"], 
            economic_limit
        )
        
        # Calculate time to economic limit
        exp_pred = prediction_result.get("ExponentialPrediction", [])
        if exp_pred:
            days_to_limit = len(exp_pred)
            months_to_limit = days_to_limit / 30
            
            # Generate interactive URL
            params["elr"] = economic_limit
            interactive_url = self.url_generator.generate_url("prediction", params)
            
            if lang == "id":
                explanation = f"""Prediksi produksi untuk sumur {params['well']} dengan economic limit {economic_limit} BOPD:

• Estimasi waktu mencapai batas ekonomis: {months_to_limit:.1f} bulan ({days_to_limit} hari)
• Model yang digunakan: Exponential Decline
• Produksi saat ini: {exp_pred[0]['value']:.2f} BOPD
• Produksi akhir: {economic_limit} BOPD

Rekomendasi: {"Pertimbangkan intervensi sumur (workover/stimulation) dalam 6 bulan ke depan." if months_to_limit < 12 else "Sumur masih dalam kondisi produktif untuk jangka menengah."}"""
            else:
                explanation = f"""Production prediction for well {params['well']} with economic limit {economic_limit} BOPD:

• Estimated time to economic limit: {months_to_limit:.1f} months ({days_to_limit} days)
• Model used: Exponential Decline
• Current production: {exp_pred[0]['value']:.2f} BOPD
• Final production: {economic_limit} BOPD

Recommendation: {"Consider well intervention (workover/stimulation) within the next 6 months." if months_to_limit < 12 else "Well remains productive for medium term."}"""
            
            return {
                "type": "response",
                "explanation": explanation,
                "app_url": interactive_url,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "type": "response",
            "explanation": "Unable to generate prediction for the specified well.",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_compare_wells(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle well comparison queries"""
        # Get all wells
        wells = await client.get_wells()
        
        # Determine time period
        period_months = params.get("period_months", 6)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        
        # Analyze each well
        well_performances = []
        for well in wells[:5]:  # Limit to first 5 wells for performance
            try:
                # Get history
                history = await client.get_history(
                    well=well,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                
                if history:
                    df = pd.DataFrame(history)
                    avg_prod = df['Production'].mean()
                    
                    # Get DCA analysis
                    dca_result = await client.automatic_dca(well)
                    decline_rate = dca_result.get("DeclineRate", {}).get("Exponential", float('inf'))
                    
                    well_performances.append({
                        "well": well,
                        "avg_production": avg_prod,
                        "decline_rate": decline_rate,
                        "data_points": len(df)
                    })
            except:
                continue
        
        # Sort by performance (high production, low decline rate)
        well_performances.sort(key=lambda x: (-x["avg_production"], x["decline_rate"]))
        
        if well_performances:
            best_well = well_performances[0]
            
            # Create comparison visualization
            chart_base64 = self.viz_generator.create_well_comparison_chart(well_performances)
            
            if lang == "id":
                explanation = f"""Analisis performa sumur dalam {period_months} bulan terakhir:

Sumur terbaik: {best_well['well']}
• Rata-rata produksi: {best_well['avg_production']:.2f} BOPD
• Decline rate: {best_well['decline_rate']:.2f}%/tahun

Top 3 sumur berdasarkan produksi rata-rata:"""
                
                for i, well in enumerate(well_performances[:3]):
                    explanation += f"\n{i+1}. {well['well']}: {well['avg_production']:.2f} BOPD (DR: {well['decline_rate']:.2f}%/tahun)"
            else:
                explanation = f"""Well performance analysis for the last {period_months} months:

Best performing well: {best_well['well']}
• Average production: {best_well['avg_production']:.2f} BOPD
• Decline rate: {best_well['decline_rate']:.2f}%/year

Top 3 wells by average production:"""
                
                for i, well in enumerate(well_performances[:3]):
                    explanation += f"\n{i+1}. {well['well']}: {well['avg_production']:.2f} BOPD (DR: {well['decline_rate']:.2f}%/year)"
            
            # Generate URL for best well
            interactive_url = self.url_generator.generate_url("history", {"well": best_well['well']})
            
            return {
                "type": "response",
                "explanation": explanation,
                "visualization": chart_base64,
                "app_url": interactive_url,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "type": "response",
            "explanation": "Unable to compare wells due to insufficient data.",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_general_dca(self, query: str, lang: str) -> Dict:
        """Handle general DCA questions"""
        if lang == "id":
            explanation = """DCA (Decline Curve Analysis) adalah metode fundamental dalam industri minyak dan gas untuk:

1. Menganalisis tren penurunan produksi sumur
2. Memprediksi produksi masa depan
3. Menentukan cadangan yang dapat dipulihkan
4. Merencanakan intervensi sumur

Tiga model utama DCA:
• Exponential: Penurunan konstan (paling konservatif)
• Harmonic: Penurunan melambat seiring waktu
• Hyperbolic: Kombinasi keduanya (paling fleksibel)

Aplikasi DCA membantu engineer membuat keputusan operasional dan ekonomi yang tepat."""
        else:
            explanation = """DCA (Decline Curve Analysis) is a fundamental method in oil and gas industry for:

1. Analyzing well production decline trends
2. Predicting future production
3. Determining recoverable reserves
4. Planning well interventions

Three main DCA models:
• Exponential: Constant decline (most conservative)
• Harmonic: Decline slows over time
• Hyperbolic: Combination of both (most flexible)

DCA application helps engineers make informed operational and economic decisions."""
        
        return {
            "type": "response",
            "explanation": explanation,
            "app_url": DCA_APP_BASE_URL,
            "timestamp": datetime.now().isoformat()
        }