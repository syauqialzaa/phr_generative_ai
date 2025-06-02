import os
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import logging
import aiohttp
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
from langdetect import detect
from urllib.parse import urlencode

matplotlib.use('Agg')

# Import LLM provider manager
from llm_providers import LLMProviderManager

# Import Milvus Lite for vector storage
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DCA API Configuration
DCA_API_BASE_URL = os.getenv("DCA_API_URL", "https://5c959a7dff3c.ngrok.app")
DCA_APP_BASE_URL = "https://syauqialzaa.github.io/dca"

# Initialize LLM Provider Manager
provider_manager = LLMProviderManager()

# Vector store configuration
MILVUS_DB_PATH = "./milvus_dca.db"

# Base knowledge for DCA
DCA_KNOWLEDGE_BASE = [
    {
        "id": 1,
        "category": "dca_basics",
        "content": "DCA (Decline Curve Analysis) adalah metode untuk menganalisis penurunan produksi sumur minyak dan gas. Terdapat tiga model utama: Exponential, Harmonic, dan Hyperbolic.",
        "keywords": ["dca", "decline curve analysis", "produksi", "production", "sumur", "well"]
    },
    {
        "id": 2,
        "category": "decline_models",
        "content": "Model Exponential: q(t) = qi * exp(-D*t), Model Harmonic: q(t) = qi / (1 + D*t), Model Hyperbolic: q(t) = qi * (1 + b*D*t)^(-1/b)",
        "keywords": ["exponential", "harmonic", "hyperbolic", "decline rate", "model"]
    },
    {
        "id": 3,
        "category": "economic_limit",
        "content": "Economic limit adalah batas produksi minimum dimana sumur masih menguntungkan secara ekonomi. Biasanya berkisar 5-10 BOPD (Barrels of Oil Per Day).",
        "keywords": ["economic limit", "elr", "bopd", "minimum production", "batas ekonomis"]
    },
    {
        "id": 4,
        "category": "job_codes",
        "content": "Job codes menunjukkan intervensi sumur: PMP (pumping), PRF (perforation), STM (stimulation), WOR (workover). Intervensi dapat meningkatkan produksi.",
        "keywords": ["job code", "intervention", "workover", "stimulation", "perforation", "pumping"]
    },
    {
        "id": 5,
        "category": "api_endpoints",
        "content": "API endpoints: GET /get_wells (daftar sumur), /get_history (data historis), /automatic_dca (analisis DCA), /predict_production (prediksi produksi), /predict_ml (prediksi ML)",
        "keywords": ["api", "endpoint", "get_wells", "get_history", "automatic_dca", "predict_production", "predict_ml"]
    }
]

# DCA API Client
class DCAApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_wells(self) -> List[str]:
        async with self.session.get(f"{self.base_url}/get_wells") as response:
            data = await response.json()
            return data.get("wells", [])
    
    async def get_history(self, well: str = None, start_date: str = None, end_date: str = None) -> List[Dict]:
        payload = {}
        if well:
            payload["well"] = well
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        
        async with self.session.post(f"{self.base_url}/get_history", json=payload) as response:
            return await response.json()
    
    async def automatic_dca(self, well: str, selected_data: List[Dict] = None) -> Dict:
        payload = {"well": well}
        if selected_data:
            payload["selected_data"] = selected_data
        
        async with self.session.post(f"{self.base_url}/automatic_dca", json=payload) as response:
            return await response.json()
    
    async def predict_production(self, well: str, economic_limit: float = 5, selected_data: Dict = None) -> Dict:
        payload = {"well": well, "economic_limit": economic_limit}
        if selected_data:
            payload["selected_data"] = selected_data
        
        async with self.session.post(f"{self.base_url}/predict_production", json=payload) as response:
            return await response.json()
    
    async def predict_ml(self, elr: float = 10) -> Dict:
        payload = {"elr": elr}
        async with self.session.post(f"{self.base_url}/predict_ml", json=payload) as response:
            return await response.json()

# Vector Store Manager
class VectorStoreManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.collection_name = "dca_knowledge"
        self.collection = None
    
    def initialize(self):
        try:
            # Connect to Milvus Lite
            connections.connect(
                alias="default",
                host="127.0.0.1",
                port="19530",
                db_name=self.db_path
            )
            
            # Create collection if not exists
            if not utility.has_collection(self.collection_name):
                self._create_collection()
                self._insert_base_knowledge()
            
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            # Fallback: operate without vector store
            pass
    
    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100)
        ]
        schema = CollectionSchema(fields, "DCA knowledge base")
        collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
    
    def _insert_base_knowledge(self):
        # Simplified: Use random embeddings for now
        # In production, use proper text embeddings (e.g., sentence-transformers)
        data = []
        for item in DCA_KNOWLEDGE_BASE:
            embedding = np.random.rand(384).tolist()
            data.append([item["id"], embedding, item["content"], item["category"]])
        
        collection = Collection(self.collection_name)
        collection.insert(data)
        collection.flush()
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        if not self.collection:
            return []
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "category"]
        )
        
        return [{"content": hit.entity.get("content"), 
                 "category": hit.entity.get("category"),
                 "score": hit.score} for hit in results[0]]

# Intent Recognition and Query Processing
class DCAQueryProcessor:
    def __init__(self):
        self.intent_patterns = {
            "get_history": ["data sumur", "history", "historis", "production data", "data produksi"],
            "analyze_decline": ["decline rate", "analisis dca", "dca analysis", "penurunan produksi"],
            "predict_future": ["estimasi", "predict", "prediksi", "forecast", "future production"],
            "compare_wells": ["perbandingan", "compare", "terbaik", "best performing", "which well"],
            "economic_analysis": ["economic limit", "batas ekonomis", "waktu ekonomis", "elr"]
        }
    
    def detect_intent(self, query: str) -> str:
        query_lower = query.lower()
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        return "general_dca"
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        params = {}
        
        # Extract well code (format: PKU00001-01)
        import re
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

# Visualization Generator
class VisualizationGenerator:
    @staticmethod
    def create_production_chart(data: List[Dict], title: str = "Production History") -> str:
        """Create production chart and return as base64 encoded image"""
        try:
            df = pd.DataFrame(data)
            
            # Create figure with subplots
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot Oil Production
            ax1.plot(pd.to_datetime(df['Date']), df['Production'], 
                    'g-', label='Oil (BOPD)', linewidth=2, marker='o', markersize=4)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Oil (BOPD)', color='g')
            ax1.tick_params(axis='y', labelcolor='g')
            ax1.grid(True, alpha=0.3)
            
            # Plot Fluid if available
            if 'Fluid' in df.columns:
                ax2 = ax1.twinx()
                ax2.plot(pd.to_datetime(df['Date']), df['Fluid'], 
                        'b-', label='Fluid (BOPD)', linewidth=2, marker='s', markersize=4)
                ax2.set_ylabel('Fluid (BOPD)', color='b')
                ax2.tick_params(axis='y', labelcolor='b')
            
            # Add title and legend
            plt.title(title, fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
    
    @staticmethod
    def create_dca_analysis_chart(actual_data: List[Dict], predictions: Dict, 
                                 decline_rates: Dict) -> str:
        """Create DCA analysis chart with predictions"""
        try:
            fig = go.Figure()
            
            # Add actual data
            actual_df = pd.DataFrame(actual_data)
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(actual_df['date']),
                y=actual_df['value'],
                mode='lines+markers',
                name='Actual Data',
                line=dict(color='black', width=3)
            ))
            
            # Add predictions
            colors = {'Exponential': 'red', 'Harmonic': 'blue', 'Hyperbolic': 'green'}
            for model_name, color in colors.items():
                if model_name in predictions:
                    pred_data = predictions[model_name]
                    fig.add_trace(go.Scatter(
                        x=[d['x'] for d in pred_data],
                        y=[d['y'] for d in pred_data],
                        mode='lines',
                        name=f'{model_name} (DR: {decline_rates[model_name]}%/yr)',
                        line=dict(color=color, width=2, dash='dash')
                    ))
            
            fig.update_layout(
                title='DCA Analysis',
                xaxis_title='Date',
                yaxis_title='Production (BOPD)',
                hovermode='x unified',
                height=500
            )
            
            # Convert to base64
            buffer = BytesIO()
            fig.write_image(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating DCA chart: {e}")
            return None

# URL Generator
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

# Main DCA Assistant
class DCAAssistant:
    def __init__(self):
        self.query_processor = DCAQueryProcessor()
        self.viz_generator = VisualizationGenerator()
        self.url_generator = DCAUrlGenerator()
        self.vector_store = VectorStoreManager(MILVUS_DB_PATH)
    
    async def process_query(self, query: str, history: List[Dict] = None) -> Dict:
        """Process DCA-related query and return comprehensive response"""
        
        # Detect language
        try:
            detected_lang = detect(query)
        except:
            detected_lang = "id"  # Default to Indonesian
        
        # Check if query is DCA-related
        if not self._is_dca_related(query):
            return {
                "type": "response",
                "explanation": self._get_non_dca_response(detected_lang),
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
                "message": self._get_error_message(detected_lang, str(e)),
                "timestamp": datetime.now().isoformat()
            }
    
    def _is_dca_related(self, query: str) -> bool:
        """Check if query is related to DCA"""
        dca_keywords = [
            "dca", "decline", "produksi", "production", "sumur", "well",
            "oil", "minyak", "bopd", "economic", "ekonomis", "predict",
            "prediksi", "forecast", "analisis", "analysis", "pku"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in dca_keywords)
    
    def _get_non_dca_response(self, lang: str) -> str:
        """Get response for non-DCA queries"""
        responses = {
            "id": "Maaf, saya hanya dapat membantu dengan pertanyaan seputar DCA (Decline Curve Analysis) dan analisis produksi sumur minyak. Silakan ajukan pertanyaan terkait DCA.",
            "en": "Sorry, I can only help with questions about DCA (Decline Curve Analysis) and oil well production analysis. Please ask DCA-related questions."
        }
        return responses.get(lang, responses["en"])
    
    def _get_error_message(self, lang: str, error: str) -> str:
        """Get error message in appropriate language"""
        messages = {
            "id": f"Maaf, terjadi kesalahan saat memproses permintaan Anda: {error}",
            "en": f"Sorry, an error occurred while processing your request: {error}"
        }
        return messages.get(lang, messages["en"])
    
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
        
        # TODO: Create DCA visualization if actual data is available
        
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
        # Use LLM to generate response based on DCA knowledge
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

# Update lifespan to include vector store initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler untuk startup dan shutdown"""
    # Startup events
    logger.info("Starting up PHR Generative AI Chat Server with DCA Integration...")
    
    # Initialize LLM
    success = await provider_manager.initialize()
    if not success:
        logger.error("Failed to initialize AI service")
    else:
        provider_info = provider_manager.get_provider_info()
        logger.info(f"AI service ready: {provider_info}")
    
    # Initialize vector store
    # vector_store = VectorStoreManager(MILVUS_DB_PATH)
    # vector_store.initialize()
    
    # Initialize DCA Assistant
    app.state.dca_assistant = DCAAssistant()
    
    yield  # Aplikasi berjalan di sini
    
    # Shutdown events
    logger.info("Shutting down PHR Generative AI Chat Server...")

# Initialize FastAPI app dengan lifespan
app = FastAPI(
    title="PHR Generative AI Chat Server with DCA Integration", 
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Add these for better WebSocket support
    expose_headers=["*"],
    max_age=600,
)

# Pydantic models
class ChatMessage(BaseModel):
    question: str
    timestamp: str

class ChatResponse(BaseModel):
    type: str = "response"
    explanation: Optional[str] = None
    message: Optional[str] = None
    visualization: Optional[str] = None
    app_url: Optional[str] = None
    timestamp: str = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.chat_histories: Dict[str, List[Dict]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if client_id not in self.chat_histories:
            self.chat_histories[client_id] = []
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, client_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def add_to_history(self, client_id: str, role: str, content: str):
        if client_id not in self.chat_histories:
            self.chat_histories[client_id] = []
        
        self.chat_histories[client_id].append({
            "role": role,
            "content": content
        })
        
        # Keep only last 20 messages to manage memory
        if len(self.chat_histories[client_id]) > 20:
            self.chat_histories[client_id] = self.chat_histories[client_id][-20:]

    def get_history(self, client_id: str) -> List[Dict]:
        return self.chat_histories.get(client_id, [])

    def clear_history(self, client_id: str):
        if client_id in self.chat_histories:
            self.chat_histories[client_id] = []

manager = ConnectionManager()

# Generate response using DCA Assistant
async def generate_response(question: str, client_id: str, app_state) -> dict:
    try:
        # Get chat history for context
        history = manager.get_history(client_id)
        
        # Process with DCA Assistant
        response = await app_state.dca_assistant.process_query(question, history)
        
        # Add to history
        manager.add_to_history(client_id, "user", question)
        if response.get("explanation"):
            manager.add_to_history(client_id, "assistant", response["explanation"])
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "type": "error",
            "message": f"Error generating response: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Index.html not found</h1>", status_code=404)

# Health check endpoint
@app.get("/health")
async def health_check():
    provider_info = provider_manager.get_provider_info()
    return {
        "status": "healthy" if provider_info["initialized"] else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ai_service": provider_info,
        "dca_integration": True
    }

# Get provider information
@app.get("/api/provider-info")
async def get_provider_info():
    return provider_manager.get_provider_info()

# REST API endpoint for chat
@app.post("/api/query")
async def process_query(message: ChatMessage):
    try:
        client_id = f"rest_client_{datetime.now().timestamp()}"
        
        logger.info(f"Processing query from {client_id}: {message.question}")
        
        # Access app state directly
        response = await generate_response(message.question, client_id, app.state)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing REST query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"ws_client_{datetime.now().timestamp()}"
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        provider_info = provider_manager.get_provider_info()
        provider_name = provider_info.get("provider_type", "AI").title()
        
        welcome_message = {
            "type": "response",
            "explanation": f"Selamat datang di PHR Generative AI dengan DCA Integration (powered by {provider_name})! Saya siap membantu analisis DCA dan produksi sumur Anda. Silakan ajukan pertanyaan seputar DCA dalam bahasa apapun.",
            "timestamp": datetime.now().isoformat()
        }
        await manager.send_personal_message(welcome_message, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            logger.info(f"Received WebSocket message from {client_id}: {message_data.get('question', '')}")
            
            # Process the question - Access app state through the app instance
            response = await generate_response(
                message_data.get("question", ""), 
                client_id,
                app.state  # Changed from request.app.state to app.state
            )
            
            # Send response back to client
            await manager.send_personal_message(response, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        error_response = {
            "type": "error",
            "message": "Terjadi kesalahan dalam koneksi",
            "timestamp": datetime.now().isoformat()
        }
        try:
            await manager.send_personal_message(error_response, websocket)
        except:
            pass
        manager.disconnect(websocket, client_id)

# Clear chat history endpoint
@app.delete("/api/clear-history/{client_id}")
async def clear_chat_history(client_id: str):
    try:
        manager.clear_history(client_id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get chat statistics
@app.get("/api/stats")
async def get_stats():
    provider_info = provider_manager.get_provider_info()
    return {
        "active_connections": len(manager.active_connections),
        "total_chat_sessions": len(manager.chat_histories),
        "ai_service": provider_info,
        "dca_integration": True,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting PHR Generative AI Chat Server with DCA Integration on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )