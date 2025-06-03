import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

from dca_client import DCAApiClient
from dca_query_processor import DCAQueryProcessor
from dca_visualization import DCAVisualizationGenerator
from dca_url_generator import DCAUrlGenerator
from vector_store import VectorStoreManager
from config import DCA_API_BASE_URL, DCA_APP_BASE_URL, MILVUS_DB_PATH

logger = logging.getLogger(__name__)

class DCAAssistant:
    def __init__(self):
        self.query_processor = DCAQueryProcessor()
        self.viz_generator = DCAVisualizationGenerator()
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
                elif intent == "predict_ml":
                    return await self._handle_predict_ml(client, params, detected_lang)
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
        """Handle production history queries with enhanced visualization and analysis"""
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
        
        # Create comprehensive visualization
        chart_base64 = self.viz_generator.create_production_chart(
            history_data,
            f"Production History - {params.get('well', 'All Wells')}"
        )
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("history", params)
        
        # Detailed statistical analysis
        df = pd.DataFrame(history_data)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic statistics
        avg_production = df['Production'].mean()
        max_production = df['Production'].max()
        min_production = df['Production'].min()
        std_production = df['Production'].std()
        
        # Trend analysis
        df_sorted = df.sort_values('Date')
        production_change = ((df_sorted['Production'].iloc[-1] - df_sorted['Production'].iloc[0]) / 
                           df_sorted['Production'].iloc[0] * 100) if len(df_sorted) > 1 else 0
        
        # Monthly analysis if data spans multiple months
        monthly_stats = ""
        if len(df) > 30:  # If more than 30 data points
            df['Month'] = df['Date'].dt.to_period('M')
            monthly_avg = df.groupby('Month')['Production'].mean()
            best_month = monthly_avg.idxmax()
            worst_month = monthly_avg.idxmin()
            monthly_stats = f"\n• Best month: {best_month} ({monthly_avg[best_month]:.1f} BOPD)\n• Worst month: {worst_month} ({monthly_avg[worst_month]:.1f} BOPD)"
        
        # Job Code analysis
        job_analysis = ""
        if 'JobCode' in df.columns:
            job_points = df[df['JobCode'].notna() & (df['JobCode'] != '')]
            if not job_points.empty:
                job_analysis = f"\n• Interventions recorded: {len(job_points)} events\n• Latest intervention: {job_points['JobCode'].iloc[-1] if len(job_points) > 0 else 'N/A'}"
        
        if lang == "id":
            explanation = f"""Analisis Komprehensif Data Historis Sumur {params.get('well', '')}

📊 STATISTIK PRODUKSI
• Periode: {df['Date'].min().strftime('%d %b %Y')} - {df['Date'].max().strftime('%d %b %Y')}
• Total data points: {len(df)} pengukuran
• Rata-rata produksi: {avg_production:.2f} BOPD
• Produksi maksimum: {max_production:.2f} BOPD  
• Produksi minimum: {min_production:.2f} BOPD
• Standar deviasi: {std_production:.2f} BOPD
• Perubahan keseluruhan: {production_change:+.1f}%{monthly_stats}{job_analysis}

📈 ANALISIS TREN
{"Tren produksi menurun" if production_change < -5 else "Tren produksi stabil" if abs(production_change) <= 5 else "Tren produksi meningkat"} dengan variabilitas {"tinggi" if std_production > avg_production * 0.2 else "sedang" if std_production > avg_production * 0.1 else "rendah"}.

🔍 REKOMENDASI
{"Monitoring ketat diperlukan karena tren penurunan signifikan." if production_change < -10 else "Pertahankan monitoring rutin dan evaluasi berkala." if abs(production_change) <= 10 else "Kondisi sumur menunjukkan performa positif."}

Grafik menunjukkan tren produksi minyak (hijau) dan fluida (biru) dengan titik intervensi (oranye) jika ada."""
        else:
            explanation = f"""Comprehensive Historical Data Analysis for Well {params.get('well', '')}

📊 PRODUCTION STATISTICS  
• Period: {df['Date'].min().strftime('%d %b %Y')} - {df['Date'].max().strftime('%d %b %Y')}
• Total data points: {len(df)} measurements
• Average production: {avg_production:.2f} BOPD
• Maximum production: {max_production:.2f} BOPD
• Minimum production: {min_production:.2f} BOPD  
• Standard deviation: {std_production:.2f} BOPD
• Overall change: {production_change:+.1f}%{monthly_stats}{job_analysis}

📈 TREND ANALYSIS
{"Declining production trend" if production_change < -5 else "Stable production trend" if abs(production_change) <= 5 else "Increasing production trend"} with {"high" if std_production > avg_production * 0.2 else "moderate" if std_production > avg_production * 0.1 else "low"} variability.

🔍 RECOMMENDATIONS  
{"Close monitoring required due to significant decline." if production_change < -10 else "Maintain routine monitoring and periodic evaluation." if abs(production_change) <= 10 else "Well condition shows positive performance."}

Chart shows oil (green) and fluid (blue) production trends with intervention points (orange) if any."""
        
        return {
            "type": "response",
            "explanation": explanation,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_analyze_decline(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle DCA analysis queries with comprehensive visualization"""
        if not params.get("well"):
            return {
                "type": "response",
                "explanation": "Please specify a well code for DCA analysis.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get DCA analysis
        dca_result = await client.automatic_dca(params["well"])
        
        if not dca_result or 'DeclineRate' not in dca_result:
            return {
                "type": "response",
                "explanation": "Unable to perform DCA analysis for the specified well.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Create DCA visualization
        chart_base64 = self.viz_generator.create_dca_analysis_chart(dca_result)
        
        # Extract detailed analysis
        decline_rates = dca_result.get("DeclineRate", {})
        exp_params = dca_result.get("Exponential", [])
        harm_params = dca_result.get("Harmonic", [])
        hyper_params = dca_result.get("Hyperbolic", [])
        actual_data = dca_result.get("ActualData", [])
        
        # Calculate model quality metrics
        qi_exp = exp_params[0] if exp_params else 0
        qi_harm = harm_params[0] if harm_params else 0
        qi_hyper = hyper_params[0] if hyper_params else 0
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("dca", params)
        
        # Model recommendations
        exp_rate = decline_rates.get('Exponential', 0)
        harm_rate = decline_rates.get('Harmonic', 0)
        hyper_rate = decline_rates.get('Hyperbolic', 0)
        
        # Determine best model based on decline rates and reservoir characteristics
        if exp_rate < 20:
            recommended_model = "Exponential"
            model_reason = "low decline rate suggests exponential behavior"
        elif harm_rate > 50:
            recommended_model = "Harmonic"
            model_reason = "high decline rate typical of harmonic decline"
        else:
            recommended_model = "Hyperbolic"
            model_reason = "moderate decline rate fits hyperbolic model"
        
        if lang == "id":
            explanation = f"""Analisis DCA Komprehensif untuk Sumur {params['well']}

📊 PARAMETER MODEL DCA
• Model Exponential: Qi = {qi_exp:.1f} BOPD, DR = {exp_rate:.2f}%/tahun
• Model Harmonic: Qi = {qi_harm:.1f} BOPD, DR = {harm_rate:.2f}%/tahun  
• Model Hyperbolic: Qi = {qi_hyper:.1f} BOPD, DR = {hyper_rate:.2f}%/tahun

📈 KARAKTERISTIK DECLINE
• Exponential: Penurunan konstan, estimasi konservatif
• Harmonic: Penurunan melambat seiring waktu, optimis untuk long-term
• Hyperbolic: Kombinasi fleksibel, umum digunakan untuk tight reservoirs

🎯 MODEL REKOMENDASI
Model {recommended_model} direkomendasikan karena {model_reason}.

⚡ KLASIFIKASI DECLINE RATE
{exp_rate:.1f}%/tahun = {"Sangat Rendah" if exp_rate < 10 else "Rendah" if exp_rate < 20 else "Sedang" if exp_rate < 40 else "Tinggi" if exp_rate < 60 else "Sangat Tinggi"}

🔍 IMPLIKASI OPERASIONAL
{"Sumur dalam kondisi sangat baik, monitoring rutin cukup." if exp_rate < 15 else "Sumur stabil, evaluasi berkala diperlukan." if exp_rate < 30 else "Pertimbangkan intervensi dalam 1-2 tahun." if exp_rate < 50 else "Intervensi segera diperlukan untuk mempertahankan produksi."}

Data periode: {len(actual_data)} titik pengukuran dari {dca_result.get('StartDate', 'N/A')} hingga {dca_result.get('EndDate', 'N/A')}."""
        else:
            explanation = f"""Comprehensive DCA Analysis for Well {params['well']}

📊 DCA MODEL PARAMETERS
• Exponential Model: Qi = {qi_exp:.1f} BOPD, DR = {exp_rate:.2f}%/year
• Harmonic Model: Qi = {qi_harm:.1f} BOPD, DR = {harm_rate:.2f}%/year
• Hyperbolic Model: Qi = {qi_hyper:.1f} BOPD, DR = {hyper_rate:.2f}%/year

📈 DECLINE CHARACTERISTICS
• Exponential: Constant decline, conservative estimates
• Harmonic: Decline slows over time, optimistic for long-term
• Hyperbolic: Flexible combination, common for tight reservoirs

🎯 RECOMMENDED MODEL
{recommended_model} model is recommended because {model_reason}.

⚡ DECLINE RATE CLASSIFICATION
{exp_rate:.1f}%/year = {"Very Low" if exp_rate < 10 else "Low" if exp_rate < 20 else "Moderate" if exp_rate < 40 else "High" if exp_rate < 60 else "Very High"}

🔍 OPERATIONAL IMPLICATIONS
{"Well in excellent condition, routine monitoring sufficient." if exp_rate < 15 else "Well stable, periodic evaluation required." if exp_rate < 30 else "Consider intervention within 1-2 years." if exp_rate < 50 else "Immediate intervention needed to maintain production."}

Analysis period: {len(actual_data)} measurement points from {dca_result.get('StartDate', 'N/A')} to {dca_result.get('EndDate', 'N/A')}."""
        
        return {
            "type": "response",
            "explanation": explanation,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_predict_future(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle production prediction queries with detailed forecasting"""
        if not params.get("well"):
            return {
                "type": "response",
                "explanation": "Please specify a well code for prediction.",
                "timestamp": datetime.now().isoformat()
            }
        
        # First run DCA analysis to get the baseline
        dca_result = await client.automatic_dca(params["well"])
        if not dca_result:
            return {
                "type": "response", 
                "explanation": "Unable to perform DCA analysis required for prediction.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get predictions
        economic_limit = params.get("economic_limit", 5)
        prediction_result = await client.predict_production(
            params["well"], 
            economic_limit
        )
        
        if not prediction_result:
            return {
                "type": "response",
                "explanation": "Unable to generate predictions for the specified well.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Create prediction visualization
        chart_base64 = self.viz_generator.create_production_prediction_chart(
            prediction_result, params["well"]
        )
        
        # Detailed analysis of predictions
        exp_pred = prediction_result.get("ExponentialPrediction", [])
        harm_pred = prediction_result.get("HarmonicPrediction", [])
        hyper_pred = prediction_result.get("HyperbolicPrediction", [])
        
        # Generate interactive URL
        params["elr"] = economic_limit
        interactive_url = self.url_generator.generate_url("prediction", params)
        
        if exp_pred:
            start_prod = exp_pred[0]['value']
            days_to_limit = len(exp_pred)
            months_to_limit = days_to_limit / 30
            years_to_limit = days_to_limit / 365
            
            # Calculate cumulative production
            total_production = sum([p['value'] for p in exp_pred])
            avg_daily_prod = total_production / days_to_limit if days_to_limit > 0 else 0
            
            # Economic analysis
            oil_price = 70  # Assumed oil price USD/barrel
            estimated_revenue = total_production * oil_price
            
            # Compare models
            model_comparison = ""
            if harm_pred and hyper_pred:
                harm_days = len(harm_pred)
                hyper_days = len(hyper_pred)
                model_comparison = f"\n• Harmonic: {harm_days/30:.1f} months to ELR\n• Hyperbolic: {hyper_days/30:.1f} months to ELR"
            
            if lang == "id":
                explanation = f"""Prediksi Produksi Komprehensif - Sumur {params['well']}

📈 PREDIKSI EXPONENTIAL (KONSERVATIF)
• Produksi awal: {start_prod:.2f} BOPD
• Economic limit: {economic_limit} BOPD
• Waktu mencapai ELR: {months_to_limit:.1f} bulan ({years_to_limit:.2f} tahun)
• Total hari produksi: {days_to_limit} hari

📊 ANALISIS PRODUKSI
• Rata-rata harian: {avg_daily_prod:.1f} BOPD
• Estimasi total produksi: {total_production:,.0f} barrels
• Estimasi revenue (@ $70/bbl): ${estimated_revenue:,.0f}

⚖️ PERBANDINGAN MODEL{model_comparison}

🔍 REKOMENDASI STRATEGIS
{"⚠️ URGENT: Pertimbangkan workover/stimulation dalam 3 bulan untuk mempertahankan produksi ekonomis." if months_to_limit < 6 else "📋 MONITORING: Rencanakan intervensi dalam 6-12 bulan ke depan." if months_to_limit < 12 else "✅ STABIL: Sumur masih produktif untuk jangka menengah, monitoring rutin." if months_to_limit < 24 else "🎯 OPTIMAL: Kondisi sumur sangat baik untuk jangka panjang."}

💰 ANALISIS EKONOMI
{"Revenue potensial rendah, evaluasi cost-benefit intervensi." if estimated_revenue < 100000 else "Revenue memadai untuk mendukung operasi rutin." if estimated_revenue < 500000 else "Revenue tinggi, prioritaskan optimasi produksi."}"""
            else:
                explanation = f"""Comprehensive Production Prediction - Well {params['well']}

📈 EXPONENTIAL PREDICTION (CONSERVATIVE)
• Initial production: {start_prod:.2f} BOPD
• Economic limit: {economic_limit} BOPD  
• Time to ELR: {months_to_limit:.1f} months ({years_to_limit:.2f} years)
• Total production days: {days_to_limit} days

📊 PRODUCTION ANALYSIS
• Daily average: {avg_daily_prod:.1f} BOPD
• Estimated total production: {total_production:,.0f} barrels
• Estimated revenue (@ $70/bbl): ${estimated_revenue:,.0f}

⚖️ MODEL COMPARISON{model_comparison}

🔍 STRATEGIC RECOMMENDATIONS
{"⚠️ URGENT: Consider workover/stimulation within 3 months to maintain economic production." if months_to_limit < 6 else "📋 MONITORING: Plan intervention within 6-12 months." if months_to_limit < 12 else "✅ STABLE: Well remains productive for medium term, routine monitoring." if months_to_limit < 24 else "🎯 OPTIMAL: Well condition excellent for long term."}

💰 ECONOMIC ANALYSIS
{"Low revenue potential, evaluate cost-benefit of intervention." if estimated_revenue < 100000 else "Adequate revenue to support routine operations." if estimated_revenue < 500000 else "High revenue potential, prioritize production optimization."}"""
            
            return {
                "type": "response",
                "explanation": explanation,
                "visualization": chart_base64,
                "app_url": interactive_url,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "type": "response",
            "explanation": "Unable to generate detailed prediction analysis.",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_predict_ml(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle ML prediction queries"""
        economic_limit = params.get("economic_limit", 10.0)
        
        # Get ML predictions
        ml_result = await client.predict_ml(economic_limit)
        
        if not ml_result:
            return {
                "type": "response",
                "explanation": "Unable to generate ML predictions.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Create ML visualization
        chart_base64 = self.viz_generator.create_ml_prediction_chart(ml_result)
        
        # Analyze ML results
        actual = ml_result.get('actual', [])
        predicted = ml_result.get('predicted', [])
        extended_prediction = ml_result.get('extended_prediction', [])
        dates_extended = ml_result.get('dates_extended', [])
        
        # Calculate model performance
        if actual and predicted:
            mse = sum([(a - p)**2 for a, p in zip(actual, predicted)]) / len(actual)
            rmse = mse ** 0.5
            mae = sum([abs(a - p) for a, p in zip(actual, predicted)]) / len(actual)
            
            # Calculate R²
            actual_mean = sum(actual) / len(actual)
            ss_tot = sum([(a - actual_mean)**2 for a in actual])
            ss_res = sum([(a - p)**2 for a, p in zip(actual, predicted)])
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Future analysis
            future_analysis = ""
            if extended_prediction and dates_extended:
                days_to_elr = len(extended_prediction)
                final_production = extended_prediction[-1]
                total_future_production = sum(extended_prediction)
                
                future_analysis = f"\n\n📅 PREDIKSI MASA DEPAN\n• Durasi hingga ELR: {days_to_elr} hari ({days_to_elr/30:.1f} bulan)\n• Produksi akhir: {final_production:.1f} BOPD\n• Total produksi future: {total_future_production:,.0f} barrels"
        
        if lang == "id":
            explanation = f"""Prediksi Machine Learning untuk Produksi Sumur

🤖 PERFORMA MODEL ML
• Root Mean Square Error (RMSE): {rmse:.2f} BOPD
• Mean Absolute Error (MAE): {mae:.2f} BOPD  
• Coefficient of Determination (R²): {r_squared:.3f}
• Kualitas model: {"Sangat Baik" if r_squared > 0.9 else "Baik" if r_squared > 0.8 else "Cukup" if r_squared > 0.7 else "Perlu Perbaikan"}

📊 ANALISIS DATA
• Data historis: {len(actual)} titik pengukuran
• Akurasi prediksi: {(r_squared * 100):.1f}%
• Economic Limit Rate: {economic_limit} BOPD{future_analysis}

🔬 KEUNGGULAN ML
• Mempertimbangkan pola non-linear kompleks
• Adaptif terhadap variasi operasional
• Mengintegrasikan multiple variabel reservoir

⚠️ CATATAN PENTING
Model ML memberikan prediksi berdasarkan historical pattern. Perubahan kondisi operasional atau reservoir dapat mempengaruhi akurasi."""
        else:
            explanation = f"""Machine Learning Production Prediction

🤖 ML MODEL PERFORMANCE
• Root Mean Square Error (RMSE): {rmse:.2f} BOPD
• Mean Absolute Error (MAE): {mae:.2f} BOPD
• Coefficient of Determination (R²): {r_squared:.3f}  
• Model quality: {"Excellent" if r_squared > 0.9 else "Good" if r_squared > 0.8 else "Fair" if r_squared > 0.7 else "Needs Improvement"}

📊 DATA ANALYSIS
• Historical data: {len(actual)} measurement points
• Prediction accuracy: {(r_squared * 100):.1f}%
• Economic Limit Rate: {economic_limit} BOPD{future_analysis}

🔬 ML ADVANTAGES
• Considers complex non-linear patterns
• Adaptive to operational variations
• Integrates multiple reservoir variables

⚠️ IMPORTANT NOTE
ML model provides predictions based on historical patterns. Changes in operational or reservoir conditions may affect accuracy."""
        
        return {
            "type": "response",
            "explanation": explanation,
            "visualization": chart_base64,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_compare_wells(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle well comparison queries with comprehensive analysis"""
        # Get all wells
        wells = await client.get_wells()
        
        if not wells:
            return {
                "type": "response",
                "explanation": "No wells available for comparison.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Determine time period
        period_months = params.get("period_months", 6)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        
        # Analyze each well
        well_performances = []
        for well in wells[:10]:  # Limit to first 10 wells for performance
            try:
                # Get history
                history = await client.get_history(
                    well=well,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                
                if history and len(history) > 5:  # Need sufficient data
                    df = pd.DataFrame(history)
                    avg_prod = df['Production'].mean()
                    max_prod = df['Production'].max()
                    min_prod = df['Production'].min()
                    std_prod = df['Production'].std()
                    
                    # Get DCA analysis for decline rate
                    try:
                        dca_result = await client.automatic_dca(well)
                        decline_rate = dca_result.get("DeclineRate", {}).get("Exponential", float('inf'))
                    except:
                        decline_rate = float('inf')
                    
                    # Calculate performance score
                    performance_score = avg_prod / (decline_rate + 1) if decline_rate != float('inf') else avg_prod
                    
                    well_performances.append({
                        "well": well,
                        "avg_production": avg_prod,
                        "max_production": max_prod,
                        "min_production": min_prod,
                        "std_production": std_prod,
                        "decline_rate": decline_rate if decline_rate != float('inf') else 999,
                        "data_points": len(df),
                        "performance_score": performance_score
                    })
            except Exception as e:
                logger.warning(f"Error analyzing well {well}: {e}")
                continue
        
        if not well_performances:
            return {
                "type": "response",
                "explanation": "Insufficient data for well comparison analysis.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Sort by performance score (high production, low decline rate)
        well_performances.sort(key=lambda x: x["performance_score"], reverse=True)
        
        # Create comparison visualization
        chart_base64 = self.viz_generator.create_well_comparison_chart(well_performances)
        
        # Detailed analysis
        best_well = well_performances[0]
        worst_well = well_performances[-1]
        
        # Calculate fleet statistics
        fleet_avg_prod = sum([w["avg_production"] for w in well_performances]) / len(well_performances)
        fleet_avg_decline = sum([w["decline_rate"] for w in well_performances if w["decline_rate"] < 999]) / len([w for w in well_performances if w["decline_rate"] < 999])
        
        # Categorize wells
        high_performers = [w for w in well_performances if w["avg_production"] > fleet_avg_prod * 1.2]
        low_performers = [w for w in well_performances if w["avg_production"] < fleet_avg_prod * 0.8]
        stable_wells = [w for w in well_performances if w["decline_rate"] < fleet_avg_decline]
        
        if lang == "id":
            explanation = f"""Analisis Perbandingan Performa Sumur ({period_months} bulan terakhir)

🏆 SUMUR TERBAIK: {best_well['well']}
• Produksi rata-rata: {best_well['avg_production']:.2f} BOPD
• Produksi maksimum: {best_well['max_production']:.2f} BOPD
• Decline rate: {best_well['decline_rate']:.2f}%/tahun
• Variabilitas: {best_well['std_production']:.2f} BOPD
• Score performa: {best_well['performance_score']:.1f}

📊 STATISTIK FLEET
• Total sumur dianalisis: {len(well_performances)}
• Rata-rata produksi fleet: {fleet_avg_prod:.2f} BOPD
• Rata-rata decline rate: {fleet_avg_decline:.2f}%/tahun
• High performers: {len(high_performers)} sumur (>{fleet_avg_prod*1.2:.1f} BOPD)
• Low performers: {len(low_performers)} sumur (<{fleet_avg_prod*0.8:.1f} BOPD)

🎯 TOP 5 SUMUR BERDASARKAN PERFORMA:"""
            
            for i, well in enumerate(well_performances[:5]):
                explanation += f"\n{i+1}. {well['well']}: {well['avg_production']:.1f} BOPD (DR: {well['decline_rate']:.1f}%/tahun)"
            
            explanation += f"""

⚠️ SUMUR PERLU PERHATIAN: {worst_well['well']}
• Produksi: {worst_well['avg_production']:.1f} BOPD
• Decline rate: {worst_well['decline_rate']:.1f}%/tahun

🔍 REKOMENDASI
• Prioritaskan optimasi pada {len(low_performers)} sumur underperforming
• Pertahankan strategi operasi untuk {len(high_performers)} sumur high-performing  
• Monitor {len(stable_wells)} sumur dengan decline rate stabil"""
        else:
            explanation = f"""Well Performance Comparison Analysis (Last {period_months} months)

🏆 BEST PERFORMING WELL: {best_well['well']}
• Average production: {best_well['avg_production']:.2f} BOPD
• Maximum production: {best_well['max_production']:.2f} BOPD
• Decline rate: {best_well['decline_rate']:.2f}%/year
• Variability: {best_well['std_production']:.2f} BOPD  
• Performance score: {best_well['performance_score']:.1f}

📊 FLEET STATISTICS
• Total wells analyzed: {len(well_performances)}
• Fleet average production: {fleet_avg_prod:.2f} BOPD
• Fleet average decline rate: {fleet_avg_decline:.2f}%/year
• High performers: {len(high_performers)} wells (>{fleet_avg_prod*1.2:.1f} BOPD)
• Low performers: {len(low_performers)} wells (<{fleet_avg_prod*0.8:.1f} BOPD)

🎯 TOP 5 WELLS BY PERFORMANCE:"""
            
            for i, well in enumerate(well_performances[:5]):
                explanation += f"\n{i+1}. {well['well']}: {well['avg_production']:.1f} BOPD (DR: {well['decline_rate']:.1f}%/year)"
            
            explanation += f"""

⚠️ WELL NEEDS ATTENTION: {worst_well['well']}
• Production: {worst_well['avg_production']:.1f} BOPD
• Decline rate: {worst_well['decline_rate']:.1f}%/year

🔍 RECOMMENDATIONS
• Prioritize optimization for {len(low_performers)} underperforming wells
• Maintain operation strategy for {len(high_performers)} high-performing wells
• Monitor {len(stable_wells)} wells with stable decline rates"""
        
        # Generate URL for best well
        interactive_url = self.url_generator.generate_url("history", {"well": best_well['well']})
        
        return {
            "type": "response",
            "explanation": explanation,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_general_dca(self, query: str, lang: str) -> Dict:
        """Handle general DCA questions with comprehensive information"""
        if lang == "id":
            explanation = """Decline Curve Analysis (DCA) - Panduan Komprehensif

📚 DEFINISI DCA
DCA adalah metode fundamental dalam industri minyak dan gas untuk menganalisis tren penurunan produksi sumur dan memprediksi performa masa depan berdasarkan data historis.

🎯 APLIKASI UTAMA
1. **Analisis Tren Produksi** - Memahami pola penurunan alamiah
2. **Prediksi Masa Depan** - Estimasi produksi hingga batas ekonomis  
3. **Estimasi Cadangan** - Menghitung recoverable reserves
4. **Perencanaan Intervensi** - Timing optimal untuk workover/stimulation
5. **Evaluasi Ekonomi** - NPV dan analisis investasi

📊 TIGA MODEL UTAMA DCA

**🔴 EXPONENTIAL DECLINE**
• Formula: q(t) = qi × e^(-Dt)
• Karakteristik: Penurunan konstan per satuan waktu
• Aplikasi: Sumur mature, conventional reservoirs
• Keunggulan: Konservatif, mudah dihitung
• Limitasi: Terlalu pesimis untuk unconventional

**🔵 HARMONIC DECLINE**  
• Formula: q(t) = qi / (1 + Dt)
• Karakteristik: Penurunan melambat seiring waktu
• Aplikasi: Tight gas, fractured reservoirs
• Keunggulan: Optimis untuk long-term production
• Limitasi: Bisa terlalu optimis di tahap awal

**🟢 HYPERBOLIC DECLINE**
• Formula: q(t) = qi / (1 + bDt)^(1/b)
• Karakteristik: Fleksibel, transisi exponential ke harmonic
• Parameter: b-factor (0 < b < 1)
• Aplikasi: Shale gas/oil, horizontal wells
• Keunggulan: Paling realistis untuk unconventional

🔍 PARAMETER KUNCI
• **qi (Initial Rate)**: Produksi awal pada t=0
• **D (Decline Rate)**: Laju penurunan (%/tahun)  
• **b (Hyperbolic Exponent)**: Faktor kelengkungan kurva
• **Economic Limit**: Batas minimum produksi ekonomis

⚡ KLASIFIKASI DECLINE RATE
• < 10%/tahun: Sangat Rendah (Excellent)
• 10-20%/tahun: Rendah (Good)  
• 20-40%/tahun: Sedang (Fair)
• 40-60%/tahun: Tinggi (Poor)
• > 60%/tahun: Sangat Tinggi (Critical)

🛠️ IMPLEMENTASI PRAKTIS
1. **Data Quality**: Minimum 6-12 titik data stabil
2. **Model Selection**: Berdasarkan reservoir type dan R²  
3. **Validation**: Cross-check dengan geological/engineering data
4. **Updates**: Review berkala dengan data terbaru
5. **Integration**: Kombinasi dengan reservoir simulation

💡 TIPS BEST PRACTICE
• Gunakan data post-transient untuk DCA
• Pertimbangkan efek intervensi sumur
• Validasi dengan offset wells
• Kombinasikan multiple scenarios
• Regular model updates dan benchmarking"""
        else:
            explanation = """Decline Curve Analysis (DCA) - Comprehensive Guide

📚 DCA DEFINITION
DCA is a fundamental method in oil and gas industry for analyzing well production decline trends and predicting future performance based on historical data.

🎯 MAIN APPLICATIONS
1. **Production Trend Analysis** - Understanding natural decline patterns
2. **Future Prediction** - Estimating production to economic limit
3. **Reserve Estimation** - Calculating recoverable reserves  
4. **Intervention Planning** - Optimal timing for workover/stimulation
5. **Economic Evaluation** - NPV and investment analysis

📊 THREE MAIN DCA MODELS

**🔴 EXPONENTIAL DECLINE**
• Formula: q(t) = qi × e^(-Dt)
• Characteristics: Constant decline per unit time
• Application: Mature wells, conventional reservoirs
• Advantages: Conservative, easy to calculate
• Limitations: Too pessimistic for unconventional

**🔵 HARMONIC DECLINE**
• Formula: q(t) = qi / (1 + Dt)  
• Characteristics: Decline slows over time
• Application: Tight gas, fractured reservoirs
• Advantages: Optimistic for long-term production
• Limitations: May be too optimistic early on

**🟢 HYPERBOLIC DECLINE**
• Formula: q(t) = qi / (1 + bDt)^(1/b)
• Characteristics: Flexible, transitions exponential to harmonic
• Parameter: b-factor (0 < b < 1)
• Application: Shale gas/oil, horizontal wells
• Advantages: Most realistic for unconventional

🔍 KEY PARAMETERS
• **qi (Initial Rate)**: Initial production at t=0
• **D (Decline Rate)**: Rate of decline (%/year)
• **b (Hyperbolic Exponent)**: Curve curvature factor
• **Economic Limit**: Minimum economic production threshold

⚡ DECLINE RATE CLASSIFICATION
• < 10%/year: Very Low (Excellent)
• 10-20%/year: Low (Good)
• 20-40%/year: Moderate (Fair)  
• 40-60%/year: High (Poor)
• > 60%/year: Very High (Critical)

🛠️ PRACTICAL IMPLEMENTATION
1. **Data Quality**: Minimum 6-12 stable data points
2. **Model Selection**: Based on reservoir type and R²
3. **Validation**: Cross-check with geological/engineering data
4. **Updates**: Periodic review with latest data
5. **Integration**: Combine with reservoir simulation

💡 BEST PRACTICE TIPS
• Use post-transient data for DCA
• Consider well intervention effects
• Validate with offset wells
• Combine multiple scenarios
• Regular model updates and benchmarking"""
        
        return {
            "type": "response",
            "explanation": explanation,
            "app_url": DCA_APP_BASE_URL,
            "timestamp": datetime.now().isoformat()
        }