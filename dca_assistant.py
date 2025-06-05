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
        
        # Log extracted parameters for debugging
        logger.info(f"Query: {query}")
        logger.info(f"Intent: {intent}")
        logger.info(f"Extracted params: {params}")
        
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
        """Handle production history queries with CORRECTED date parameter handling"""
        
        # ===== CRITICAL FIX: Ensure date parameters are passed correctly =====
        well_code = params.get("well")
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        single_date = params.get("date")
        
        # Log the parameters being used
        logger.info(f"History request - Well: {well_code}, Start: {start_date}, End: {end_date}, Single: {single_date}")
        
        # If single date provided but no range, use it as end date and set start date to 1 year earlier
        if single_date and not start_date and not end_date:
            end_date = single_date
            try:
                end_dt = datetime.strptime(single_date, "%Y-%m-%d")
                start_dt = end_dt - timedelta(days=365)
                start_date = start_dt.strftime("%Y-%m-%d")
            except:
                logger.warning(f"Could not parse single date: {single_date}")
        
        # Validate date range
        if start_date and end_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
                # Ensure start_date is before end_date
                if start_dt > end_dt:
                    start_date, end_date = end_date, start_date
                    logger.info(f"Swapped dates - Start: {start_date}, End: {end_date}")
                
                # Check for reasonable date range (not more than 10 years)
                date_diff = end_dt - start_dt
                if date_diff.days > 3650:  # 10 years
                    logger.warning(f"Date range too large: {date_diff.days} days")
                    
            except ValueError as e:
                logger.error(f"Invalid date format: {e}")
                # Reset to None to get all available data
                start_date = None
                end_date = None
        
        # ===== MAKE API CALL WITH CORRECT PARAMETERS =====
        try:
            history_data = await client.get_history(
                well=well_code,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {
                "type": "error",
                "message": f"Failed to fetch data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        if not history_data:
            # Try without date filter if no data found with dates
            if start_date or end_date:
                logger.info("No data found with date filter, trying without dates")
                try:
                    history_data = await client.get_history(well=well_code)
                except:
                    pass
            
            if not history_data:
                return {
                    "type": "response",
                    "explanation": f"No data found for well {well_code}" + (f" between {start_date} and {end_date}" if start_date and end_date else ""),
                    "timestamp": datetime.now().isoformat()
                }
        
        # ===== FILTER DATA POST-API IF NEEDED =====
        if history_data and (start_date or end_date):
            filtered_data = []
            for record in history_data:
                try:
                    record_date = datetime.strptime(record['Date'], "%Y-%m-%d")
                    
                    # Apply date filters
                    include_record = True
                    if start_date:
                        filter_start = datetime.strptime(start_date, "%Y-%m-%d")
                        if record_date < filter_start:
                            include_record = False
                    
                    if end_date and include_record:
                        filter_end = datetime.strptime(end_date, "%Y-%m-%d")
                        if record_date > filter_end:
                            include_record = False
                    
                    if include_record:
                        filtered_data.append(record)
                        
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid record: {e}")
                    continue
            
            if filtered_data:
                history_data = filtered_data
                logger.info(f"Filtered data: {len(filtered_data)} records")
            else:
                logger.warning("No data after filtering")
        
        # Create comprehensive visualization with correct title
        title_parts = ["Production History"]
        if well_code:
            title_parts.append(f"- {well_code}")
        if start_date and end_date:
            start_formatted = datetime.strptime(start_date, "%Y-%m-%d").strftime("%b %Y")
            end_formatted = datetime.strptime(end_date, "%Y-%m-%d").strftime("%b %Y")
            title_parts.append(f"({start_formatted} - {end_formatted})")
        
        chart_title = " ".join(title_parts)
        
        chart_base64 = self.viz_generator.create_production_chart(
            history_data,
            chart_title
        )
        
        # Generate interactive URL with correct parameters
        url_params = {}
        if well_code:
            url_params["well"] = well_code
        if start_date:
            url_params["start_date"] = start_date
        if end_date:
            url_params["end_date"] = end_date
            
        interactive_url = self.url_generator.generate_url("history", url_params)
        
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
        
        # Format date range for display
        date_range_display = ""
        if start_date and end_date:
            start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime('%d %b %Y')
            end_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime('%d %b %Y')
            date_range_display = f"Periode: {start_fmt} - {end_fmt}"
        else:
            date_range_display = f"Periode: {df['Date'].min().strftime('%d %b %Y')} - {df['Date'].max().strftime('%d %b %Y')}"
        
        if lang == "id":
            explanation = f"""✅ **Analisis Data Historis Berhasil untuk Sumur {params.get('well', 'Unknown')}**

📊 **STATISTIK PRODUKSI**
• {date_range_display}
• Total data points: {len(df)} pengukuran
• Rata-rata produksi: {avg_production:.2f} BOPD
• Produksi maksimum: {max_production:.2f} BOPD  
• Produksi minimum: {min_production:.2f} BOPD
• Standar deviasi: {std_production:.2f} BOPD
• Perubahan keseluruhan: {production_change:+.1f}%{monthly_stats}{job_analysis}

📈 **ANALISIS TREN**
{"Tren produksi menurun" if production_change < -5 else "Tren produksi stabil" if abs(production_change) <= 5 else "Tren produksi meningkat"} dengan variabilitas {"tinggi" if std_production > avg_production * 0.2 else "sedang" if std_production > avg_production * 0.1 else "rendah"}.

🔍 **REKOMENDASI**
{"Monitoring ketat diperlukan karena tren penurunan signifikan." if production_change < -10 else "Pertahankan monitoring rutin dan evaluasi berkala." if abs(production_change) <= 10 else "Kondisi sumur menunjukkan performa positif."}

📊 **Parameter Query Yang Digunakan:**
• Well: {params.get('well', 'Tidak ditentukan')}
• Start Date: {start_date if start_date else 'Tidak ditentukan'}
• End Date: {end_date if end_date else 'Tidak ditentukan'}

Grafik menunjukkan tren produksi minyak (hijau) dan fluida (biru) dengan titik intervensi (oranye) jika ada."""
        else:
            explanation = f"""✅ **Historical Data Analysis Successful for Well {params.get('well', 'Unknown')}**

📊 **PRODUCTION STATISTICS**  
• {date_range_display}
• Total data points: {len(df)} measurements
• Average production: {avg_production:.2f} BOPD
• Maximum production: {max_production:.2f} BOPD
• Minimum production: {min_production:.2f} BOPD  
• Standard deviation: {std_production:.2f} BOPD
• Overall change: {production_change:+.1f}%{monthly_stats}{job_analysis}

📈 **TREND ANALYSIS**
{"Declining production trend" if production_change < -5 else "Stable production trend" if abs(production_change) <= 5 else "Increasing production trend"} with {"high" if std_production > avg_production * 0.2 else "moderate" if std_production > avg_production * 0.1 else "low"} variability.

🔍 **RECOMMENDATIONS**  
{"Close monitoring required due to significant decline." if production_change < -10 else "Maintain routine monitoring and periodic evaluation." if abs(production_change) <= 10 else "Well condition shows positive performance."}

📊 **Query Parameters Used:**
• Well: {params.get('well', 'Not specified')}
• Start Date: {start_date if start_date else 'Not specified'}
• End Date: {end_date if end_date else 'Not specified'}

Chart shows oil (green) and fluid (blue) production trends with intervention points (orange) if any."""
        
        return {
            "type": "response",
            "explanation": explanation,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat(),
            "query_params_used": {
                "well": well_code,
                "start_date": start_date,
                "end_date": end_date,
                "data_points": len(df)
            }
        }
    
    async def _handle_analyze_decline(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle DCA analysis queries with PROPER API parameter handling and error checking"""
        if not params.get("well"):
            return {
                "type": "response",
                "explanation": "Please specify a well code for DCA analysis.",
                "timestamp": datetime.now().isoformat()
            }
        
        well_code = params.get("well")
        
        # ===== CRITICAL FIX: Check if well exists first =====
        try:
            available_wells = await client.get_wells()
            if well_code not in available_wells:
                return {
                    "type": "error",
                    "message": f"Well {well_code} not found. Available wells: {', '.join(available_wells[:10])}{'...' if len(available_wells) > 10 else ''}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.warning(f"Could not verify well existence: {e}")
        
        # ===== PREPARE PROPER API PARAMETERS =====
        api_params = {"well": well_code}
        
        # If date range is specified, get filtered data first
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        selected_data = None
        
        if start_date or end_date:
            try:
                logger.info(f"Getting filtered data for DCA: {well_code}, {start_date} to {end_date}")
                history_data = await client.get_history(
                    well=well_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if history_data:
                    # Convert to format expected by DCA API
                    selected_data = []
                    for record in history_data:
                        selected_data.append({
                            "date": record.get("Date"),
                            "production": record.get("Production", 0),
                            "fluid": record.get("Fluid", 0) if "Fluid" in record else None
                        })
                    api_params["selected_data"] = selected_data
                    logger.info(f"Using {len(selected_data)} data points for DCA analysis")
                else:
                    logger.warning(f"No data found for date range {start_date} to {end_date}")
                    
            except Exception as e:
                logger.error(f"Error getting filtered data: {e}")
        
        # ===== MAKE DCA API CALL WITH PROPER ERROR HANDLING =====
        try:
            logger.info(f"Calling DCA API with params: {api_params}")
            dca_result = await client.automatic_dca(well_code, selected_data)
            
            if not dca_result:
                return {
                    "type": "error",
                    "message": f"DCA API returned empty result for well {well_code}",
                    "timestamp": datetime.now().isoformat()
                }
            
            if 'error' in dca_result:
                return {
                    "type": "error",
                    "message": f"DCA analysis failed: {dca_result.get('error')}",
                    "timestamp": datetime.now().isoformat()
                }
                
            if 'DeclineRate' not in dca_result:
                return {
                    "type": "error",
                    "message": f"Invalid DCA response format. Missing DeclineRate data.",
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"DCA API call failed: {e}")
            return {
                "type": "error",
                "message": f"Failed to perform DCA analysis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        # ===== CREATE VISUALIZATION WITH ACTUAL API RESPONSE =====
        try:
            chart_base64 = self.viz_generator.create_dca_analysis_chart(dca_result)
            
            if not chart_base64:
                logger.warning("Failed to create DCA chart")
                chart_base64 = None
                
        except Exception as e:
            logger.error(f"Error creating DCA visualization: {e}")
            chart_base64 = None
        
        # ===== EXTRACT AND VALIDATE DCA RESULTS =====
        decline_rates = dca_result.get("DeclineRate", {})
        exp_params = dca_result.get("Exponential", [])
        harm_params = dca_result.get("Harmonic", [])
        hyper_params = dca_result.get("Hyperbolic", [])
        actual_data = dca_result.get("ActualData", [])
        
        # Extract R-squared values if available
        r_squared_data = dca_result.get("RSquared", {})
        
        # Calculate model quality metrics with validation
        qi_exp = exp_params[0] if exp_params and len(exp_params) > 0 else 0
        qi_harm = harm_params[0] if harm_params and len(harm_params) > 0 else 0
        qi_hyper = hyper_params[0] if hyper_params and len(hyper_params) > 0 else 0
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("dca", params)
        
        # Model recommendations with validation
        exp_rate = decline_rates.get('Exponential', 0)
        harm_rate = decline_rates.get('Harmonic', 0)
        hyper_rate = decline_rates.get('Hyperbolic', 0)
        
        # Extract R-squared values
        r2_exp = r_squared_data.get('Exponential', 0)
        r2_harm = r_squared_data.get('Harmonic', 0)
        r2_hyper = r_squared_data.get('Hyperbolic', 0)
        
        # Determine best model based on R-squared and decline rates
        model_scores = {}
        if r2_exp > 0:
            model_scores['Exponential'] = r2_exp
        if r2_harm > 0:
            model_scores['Harmonic'] = r2_harm
        if r2_hyper > 0:
            model_scores['Hyperbolic'] = r2_hyper
        
        if model_scores:
            recommended_model = max(model_scores, key=model_scores.get)
            best_r2 = model_scores[recommended_model]
            model_reason = f"highest R² value ({best_r2:.3f})"
        else:
            # Fallback to decline rate logic
            if exp_rate < 20:
                recommended_model = "Exponential"
                model_reason = "low decline rate suggests exponential behavior"
            elif harm_rate > 50:
                recommended_model = "Harmonic"
                model_reason = "high decline rate typical of harmonic decline"
            else:
                recommended_model = "Hyperbolic"
                model_reason = "moderate decline rate fits hyperbolic model"
        
        # Format date range for display
        date_range_info = ""
        if start_date and end_date:
            start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime('%d %b %Y')
            end_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime('%d %b %Y')
            date_range_info = f"\n• Periode analisis: {start_fmt} - {end_fmt}"
        
        if lang == "id":
            explanation = f"""✅ **Analisis DCA Komprehensif untuk Sumur {well_code}**

📊 **PARAMETER MODEL DCA**
• Model Exponential: Qi = {qi_exp:.1f} BOPD, DR = {exp_rate:.2f}%/tahun, R² = {r2_exp:.3f}
• Model Harmonic: Qi = {qi_harm:.1f} BOPD, DR = {harm_rate:.2f}%/tahun, R² = {r2_harm:.3f}
• Model Hyperbolic: Qi = {qi_hyper:.1f} BOPD, DR = {hyper_rate:.2f}%/tahun, R² = {r2_hyper:.3f}

📈 **KARAKTERISTIK DECLINE**
• Exponential: Penurunan konstan, estimasi konservatif
• Harmonic: Penurunan melambat seiring waktu, optimis untuk long-term
• Hyperbolic: Kombinasi fleksibel, umum digunakan untuk tight reservoirs

🎯 **MODEL REKOMENDASI**
Model {recommended_model} direkomendasikan karena {model_reason}.

⚡ **KLASIFIKASI DECLINE RATE**
{exp_rate:.1f}%/tahun = {"Sangat Rendah" if exp_rate < 10 else "Rendah" if exp_rate < 20 else "Sedang" if exp_rate < 40 else "Tinggi" if exp_rate < 60 else "Sangat Tinggi"}

🔍 **KUALITAS MODEL (R-SQUARED)**
• R² > 0.90: Sangat Baik
• R² > 0.80: Baik  
• R² > 0.70: Cukup
• R² < 0.70: Perlu Perbaikan

📊 **DATA ANALYSIS**{date_range_info}
• Data points digunakan: {len(actual_data)} pengukuran
• Model terbaik: {recommended_model} (R² = {model_scores.get(recommended_model, 0):.3f})

🔍 **IMPLIKASI OPERASIONAL**
{"Sumur dalam kondisi sangat baik, monitoring rutin cukup." if exp_rate < 15 else "Sumur stabil, evaluasi berkala diperlukan." if exp_rate < 30 else "Pertimbangkan intervensi dalam 1-2 tahun." if exp_rate < 50 else "Intervensi segera diperlukan untuk mempertahankan produksi."}"""
        else:
            explanation = f"""✅ **Comprehensive DCA Analysis for Well {well_code}**

📊 **DCA MODEL PARAMETERS**
• Exponential Model: Qi = {qi_exp:.1f} BOPD, DR = {exp_rate:.2f}%/year, R² = {r2_exp:.3f}
• Harmonic Model: Qi = {qi_harm:.1f} BOPD, DR = {harm_rate:.2f}%/year, R² = {r2_harm:.3f}
• Hyperbolic Model: Qi = {qi_hyper:.1f} BOPD, DR = {hyper_rate:.2f}%/year, R² = {r2_hyper:.3f}

📈 **DECLINE CHARACTERISTICS**
• Exponential: Constant decline, conservative estimates
• Harmonic: Decline slows over time, optimistic for long-term
• Hyperbolic: Flexible combination, common for tight reservoirs

🎯 **RECOMMENDED MODEL**
{recommended_model} model is recommended because {model_reason}.

⚡ **DECLINE RATE CLASSIFICATION**
{exp_rate:.1f}%/year = {"Very Low" if exp_rate < 10 else "Low" if exp_rate < 20 else "Moderate" if exp_rate < 40 else "High" if exp_rate < 60 else "Very High"}

🔍 **MODEL QUALITY (R-SQUARED)**
• R² > 0.90: Excellent
• R² > 0.80: Good
• R² > 0.70: Fair
• R² < 0.70: Needs Improvement

📊 **DATA ANALYSIS**{date_range_info}
• Data points used: {len(actual_data)} measurements
• Best model: {recommended_model} (R² = {model_scores.get(recommended_model, 0):.3f})

🔍 **OPERATIONAL IMPLICATIONS**
{"Well in excellent condition, routine monitoring sufficient." if exp_rate < 15 else "Well stable, periodic evaluation required." if exp_rate < 30 else "Consider intervention within 1-2 years." if exp_rate < 50 else "Immediate intervention needed to maintain production."}"""
        
        return {
            "type": "response",
            "explanation": explanation,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat(),
            "dca_results": {
                "well": well_code,
                "decline_rates": decline_rates,
                "r_squared": r_squared_data,
                "recommended_model": recommended_model,
                "data_points": len(actual_data),
                "date_range": f"{start_date} to {end_date}" if start_date and end_date else "All available data"
            }
        }
    
    # [Continue with remaining methods - they need similar fixes]
    async def _handle_predict_future(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle production prediction queries with detailed forecasting and proper API handling"""
        if not params.get("well"):
            return {
                "type": "response",
                "explanation": "Please specify a well code for prediction.",
                "timestamp": datetime.now().isoformat()
            }
        
        well_code = params.get("well")
        
        # ===== CHECK WELL EXISTS =====
        try:
            available_wells = await client.get_wells()
            if well_code not in available_wells:
                return {
                    "type": "error",
                    "message": f"Well {well_code} not found. Available wells: {', '.join(available_wells[:10])}{'...' if len(available_wells) > 10 else ''}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.warning(f"Could not verify well existence: {e}")
        
        # ===== FIRST RUN DCA ANALYSIS TO GET BASELINE =====
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        selected_data = None
        
        if start_date or end_date:
            try:
                history_data = await client.get_history(
                    well=well_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if history_data:
                    selected_data = []
                    for record in history_data:
                        selected_data.append({
                            "date": record.get("Date"),
                            "production": record.get("Production", 0),
                            "fluid": record.get("Fluid", 0) if "Fluid" in record else None
                        })
            except Exception as e:
                logger.error(f"Error getting filtered data for prediction: {e}")
        
        try:
            dca_result = await client.automatic_dca(well_code, selected_data)
            if not dca_result:
                return {
                    "type": "response", 
                    "explanation": "Unable to perform DCA analysis required for prediction.",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"DCA analysis failed for prediction: {e}")
            return {
                "type": "error",
                "message": f"Failed to perform DCA analysis required for prediction: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        # ===== GET PREDICTIONS WITH PROPER PARAMETERS =====
        economic_limit = params.get("economic_limit", 5)
        
        try:
            prediction_result = await client.predict_production(
                well_code,
                economic_limit,
                selected_data
            )
            
            if not prediction_result:
                return {
                    "type": "error",
                    "message": "Unable to generate predictions for the specified well.",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Prediction API call failed: {e}")
            return {
                "type": "error",
                "message": f"Failed to generate predictions: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        # ===== CREATE PREDICTION VISUALIZATION =====
        try:
            chart_base64 = self.viz_generator.create_production_prediction_chart(
                prediction_result, well_code
            )
        except Exception as e:
            logger.error(f"Error creating prediction visualization: {e}")
            chart_base64 = None
        
        # ===== DETAILED ANALYSIS OF PREDICTIONS =====
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
            
            # Format date range info
            date_range_info = ""
            if start_date and end_date:
                start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime('%d %b %Y')
                end_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime('%d %b %Y')
                date_range_info = f"\n• Data periode: {start_fmt} - {end_fmt}"
            
            if lang == "id":
                explanation = f"""✅ **Prediksi Produksi Komprehensif - Sumur {well_code}**

📈 **PREDIKSI EXPONENTIAL (KONSERVATIF)**
• Produksi awal: {start_prod:.2f} BOPD
• Economic limit: {economic_limit} BOPD
• Waktu mencapai ELR: {months_to_limit:.1f} bulan ({years_to_limit:.2f} tahun)
• Total hari produksi: {days_to_limit} hari

📊 **ANALISIS PRODUKSI**
• Rata-rata harian: {avg_daily_prod:.1f} BOPD
• Estimasi total produksi: {total_production:,.0f} barrels
• Estimasi revenue (@ $70/bbl): ${estimated_revenue:,.0f}

⚖️ **PERBANDINGAN MODEL**{model_comparison}

📊 **PARAMETER YANG DIGUNAKAN**
• Well: {well_code}
• Economic Limit: {economic_limit} BOPD{date_range_info}
• Data points: {len(selected_data) if selected_data else 'All available'}

🔍 **REKOMENDASI STRATEGIS**
{"⚠️ URGENT: Pertimbangkan workover/stimulation dalam 3 bulan untuk mempertahankan produksi ekonomis." if months_to_limit < 6 else "📋 MONITORING: Rencanakan intervensi dalam 6-12 bulan ke depan." if months_to_limit < 12 else "✅ STABIL: Sumur masih produktif untuk jangka menengah, monitoring rutin." if months_to_limit < 24 else "🎯 OPTIMAL: Kondisi sumur sangat baik untuk jangka panjang."}

💰 **ANALISIS EKONOMI**
{"Revenue potensial rendah, evaluasi cost-benefit intervensi." if estimated_revenue < 100000 else "Revenue memadai untuk mendukung operasi rutin." if estimated_revenue < 500000 else "Revenue tinggi, prioritaskan optimasi produksi."}"""
            else:
                explanation = f"""✅ **Comprehensive Production Prediction - Well {well_code}**

📈 **EXPONENTIAL PREDICTION (CONSERVATIVE)**
• Initial production: {start_prod:.2f} BOPD
• Economic limit: {economic_limit} BOPD  
• Time to ELR: {months_to_limit:.1f} months ({years_to_limit:.2f} years)
• Total production days: {days_to_limit} days

📊 **PRODUCTION ANALYSIS**
• Daily average: {avg_daily_prod:.1f} BOPD
• Estimated total production: {total_production:,.0f} barrels
• Estimated revenue (@ $70/bbl): ${estimated_revenue:,.0f}

⚖️ **MODEL COMPARISON**{model_comparison}

📊 **PARAMETERS USED**
• Well: {well_code}
• Economic Limit: {economic_limit} BOPD{date_range_info}
• Data points: {len(selected_data) if selected_data else 'All available'}

🔍 **STRATEGIC RECOMMENDATIONS**
{"⚠️ URGENT: Consider workover/stimulation within 3 months to maintain economic production." if months_to_limit < 6 else "📋 MONITORING: Plan intervention within 6-12 months." if months_to_limit < 12 else "✅ STABLE: Well remains productive for medium term, routine monitoring." if months_to_limit < 24 else "🎯 OPTIMAL: Well condition excellent for long term."}

💰 **ECONOMIC ANALYSIS**
{"Low revenue potential, evaluate cost-benefit of intervention." if estimated_revenue < 100000 else "Adequate revenue to support routine operations." if estimated_revenue < 500000 else "High revenue potential, prioritize production optimization."}"""
            
            return {
                "type": "response",
                "explanation": explanation,
                "visualization": chart_base64,
                "app_url": interactive_url,
                "timestamp": datetime.now().isoformat(),
                "prediction_results": {
                    "well": well_code,
                    "economic_limit": economic_limit,
                    "exponential_prediction": {
                        "days_to_elr": days_to_limit,
                        "months_to_elr": months_to_limit,
                        "total_production": total_production,
                        "estimated_revenue": estimated_revenue
                    },
                    "data_range": f"{start_date} to {end_date}" if start_date and end_date else "All available data"
                }
            }
        
        return {
            "type": "error",
            "message": "Unable to generate detailed prediction analysis - no prediction data returned.",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_predict_ml(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle ML prediction queries with proper parameter handling"""
        economic_limit = params.get("economic_limit", 10.0)
        
        # ===== GET ML PREDICTIONS WITH PROPER ERROR HANDLING =====
        try:
            ml_result = await client.predict_ml(economic_limit)
            
            if not ml_result:
                return {
                    "type": "error",
                    "message": "Unable to generate ML predictions.",
                    "timestamp": datetime.now().isoformat()
                }
                
            if 'error' in ml_result:
                return {
                    "type": "error",
                    "message": f"ML prediction failed: {ml_result.get('error')}",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"ML prediction API call failed: {e}")
            return {
                "type": "error",
                "message": f"Failed to generate ML predictions: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        # ===== CREATE ML VISUALIZATION =====
        try:
            chart_base64 = self.viz_generator.create_ml_prediction_chart(ml_result)
        except Exception as e:
            logger.error(f"Error creating ML visualization: {e}")
            chart_base64 = None
        
        # ===== ANALYZE ML RESULTS =====
        actual = ml_result.get('actual', [])
        predicted = ml_result.get('predicted', [])
        extended_prediction = ml_result.get('extended_prediction', [])
        dates_extended = ml_result.get('dates_extended', [])
        confidence_interval = ml_result.get('confidence_interval', {})
        
        # Calculate model performance metrics
        performance_metrics = {}
        if actual and predicted and len(actual) == len(predicted):
            mse = sum([(a - p)**2 for a, p in zip(actual, predicted)]) / len(actual)
            rmse = mse ** 0.5
            mae = sum([abs(a - p) for a, p in zip(actual, predicted)]) / len(actual)
            
            # Calculate R²
            actual_mean = sum(actual) / len(actual)
            ss_tot = sum([(a - actual_mean)**2 for a in actual])
            ss_res = sum([(a - p)**2 for a, p in zip(actual, predicted)])
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            performance_metrics = {
                "rmse": rmse,
                "mae": mae,
                "r_squared": r_squared,
                "data_points": len(actual)
            }
            
            # Future analysis
            future_analysis = ""
            if extended_prediction and dates_extended:
                days_to_elr = len(extended_prediction)
                final_production = extended_prediction[-1]
                total_future_production = sum(extended_prediction)
                
                future_analysis = f"\n\n📅 **PREDIKSI MASA DEPAN**\n• Durasi hingga ELR: {days_to_elr} hari ({days_to_elr/30:.1f} bulan)\n• Produksi akhir: {final_production:.1f} BOPD\n• Total produksi future: {total_future_production:,.0f} barrels"
        else:
            performance_metrics = {"error": "Insufficient data for performance calculation"}
            future_analysis = ""
        
        # Add confidence interval information
        confidence_info = ""
        if confidence_interval:
            upper_bound = confidence_interval.get('upper', [])
            lower_bound = confidence_interval.get('lower', [])
            if upper_bound and lower_bound:
                avg_upper = sum(upper_bound) / len(upper_bound)
                avg_lower = sum(lower_bound) / len(lower_bound)
                confidence_info = f"\n\n🎯 **CONFIDENCE INTERVAL**\n• Upper bound average: {avg_upper:.1f} BOPD\n• Lower bound average: {avg_lower:.1f} BOPD\n• Confidence level: 95%"
        
        if lang == "id":
            explanation = f"""✅ **Prediksi Machine Learning untuk Produksi Sumur**

🤖 **PERFORMA MODEL ML**"""
            
            if "error" not in performance_metrics:
                explanation += f"""
• Root Mean Square Error (RMSE): {performance_metrics['rmse']:.2f} BOPD
• Mean Absolute Error (MAE): {performance_metrics['mae']:.2f} BOPD  
• Coefficient of Determination (R²): {performance_metrics['r_squared']:.3f}
• Kualitas model: {"Sangat Baik" if performance_metrics['r_squared'] > 0.9 else "Baik" if performance_metrics['r_squared'] > 0.8 else "Cukup" if performance_metrics['r_squared'] > 0.7 else "Perlu Perbaikan"}"""
            else:
                explanation += f"\n• Error: {performance_metrics['error']}"
            
            explanation += f"""

📊 **ANALISIS DATA**
• Data historis: {len(actual)} titik pengukuran
• Akurasi prediksi: {(performance_metrics.get('r_squared', 0) * 100):.1f}%
• Economic Limit Rate: {economic_limit} BOPD{future_analysis}{confidence_info}

🔬 **KEUNGGULAN ML**
• Mempertimbangkan pola non-linear kompleks
• Adaptif terhadap variasi operasional
• Mengintegrasikan multiple variabel reservoir
• Provides confidence intervals untuk uncertainty quantification

⚠️ **CATATAN PENTING**
Model ML memberikan prediksi berdasarkan historical pattern. Perubahan kondisi operasional atau reservoir dapat mempengaruhi akurasi."""
        else:
            explanation = f"""✅ **Machine Learning Production Prediction**

🤖 **ML MODEL PERFORMANCE**"""
            
            if "error" not in performance_metrics:
                explanation += f"""
• Root Mean Square Error (RMSE): {performance_metrics['rmse']:.2f} BOPD
• Mean Absolute Error (MAE): {performance_metrics['mae']:.2f} BOPD
• Coefficient of Determination (R²): {performance_metrics['r_squared']:.3f}  
• Model quality: {"Excellent" if performance_metrics['r_squared'] > 0.9 else "Good" if performance_metrics['r_squared'] > 0.8 else "Fair" if performance_metrics['r_squared'] > 0.7 else "Needs Improvement"}"""
            else:
                explanation += f"\n• Error: {performance_metrics['error']}"
            
            explanation += f"""

📊 **DATA ANALYSIS**
• Historical data: {len(actual)} measurement points
• Prediction accuracy: {(performance_metrics.get('r_squared', 0) * 100):.1f}%
• Economic Limit Rate: {economic_limit} BOPD{future_analysis}{confidence_info}

🔬 **ML ADVANTAGES**
• Considers complex non-linear patterns
• Adaptive to operational variations
• Integrates multiple reservoir variables
• Provides confidence intervals for uncertainty quantification

⚠️ **IMPORTANT NOTE**
ML model provides predictions based on historical patterns. Changes in operational or reservoir conditions may affect accuracy."""
        
        return {
            "type": "response",
            "explanation": explanation,
            "visualization": chart_base64,
            "timestamp": datetime.now().isoformat(),
            "ml_results": {
                "economic_limit": economic_limit,
                "performance_metrics": performance_metrics,
                "prediction_points": len(extended_prediction) if extended_prediction else 0,
                "has_confidence_interval": bool(confidence_interval)
            }
        }
    
    async def _handle_compare_wells(self, client: DCAApiClient, params: Dict, lang: str) -> Dict:
        """Handle well comparison queries with comprehensive analysis and proper error handling"""
        # ===== GET WELLS WITH ERROR HANDLING =====
        try:
            wells = await client.get_wells()
            
            if not wells:
                return {
                    "type": "error",
                    "message": "No wells available for comparison.",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to get wells list: {e}")
            return {
                "type": "error",
                "message": f"Failed to retrieve wells list: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        # ===== HANDLE SPECIFIC WELLS REQUEST =====
        specific_wells = []
        query_text = params.get("original_query", "").lower()
        
        # Extract specific well codes from query
        well_patterns = [
            r'PKU\d{5}-\d{2}',
            r'pku\d{5}-\d{2}'
        ]
        
        for pattern in well_patterns:
            import re
            found_wells = re.findall(pattern, query_text, re.IGNORECASE)
            specific_wells.extend([w.upper() for w in found_wells])
        
        # Remove duplicates while preserving order
        specific_wells = list(dict.fromkeys(specific_wells))
        
        # Use specific wells if provided, otherwise use all wells
        if specific_wells:
            wells_to_analyze = specific_wells
            logger.info(f"Analyzing specific wells: {wells_to_analyze}")
        else:
            wells_to_analyze = wells[:10]  # Limit to first 10 wells for performance
            logger.info(f"Analyzing all available wells (limited to 10): {wells_to_analyze}")
        
        # ===== DETERMINE TIME PERIOD =====
        period_months = params.get("period_months", 6)
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        
        if not start_date or not end_date:
            end_date_dt = datetime.now()
            start_date_dt = end_date_dt - timedelta(days=period_months * 30)
            start_date = start_date_dt.strftime("%Y-%m-%d")
            end_date = end_date_dt.strftime("%Y-%m-%d")
        
        logger.info(f"Comparison period: {start_date} to {end_date}")
        
        # ===== ANALYZE EACH WELL =====
        well_performances = []
        failed_wells = []
        
        for well in wells_to_analyze:
            try:
                # Get history with error handling
                history = await client.get_history(
                    well=well,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if history and len(history) > 5:  # Need sufficient data
                    df = pd.DataFrame(history)
                    
                    # Calculate basic statistics
                    avg_prod = df['Production'].mean()
                    max_prod = df['Production'].max()
                    min_prod = df['Production'].min()
                    std_prod = df['Production'].std()
                    
                    # Calculate fluid ratio if available
                    fluid_ratio = 0
                    if 'Fluid' in df.columns:
                        total_fluid = df['Fluid'].mean()
                        if total_fluid > 0:
                            fluid_ratio = (total_fluid - avg_prod) / total_fluid * 100
                    
                    # Get DCA analysis for decline rate with error handling
                    decline_rate = float('inf')
                    try:
                        dca_result = await client.automatic_dca(well)
                        if dca_result and 'DeclineRate' in dca_result:
                            decline_rate = dca_result.get("DeclineRate", {}).get("Exponential", float('inf'))
                    except Exception as dca_error:
                        logger.warning(f"DCA analysis failed for well {well}: {dca_error}")
                    
                    # Calculate performance score (high production, low decline rate)
                    if decline_rate != float('inf') and decline_rate > 0:
                        performance_score = avg_prod / (decline_rate + 1)
                    else:
                        performance_score = avg_prod
                    
                    well_performances.append({
                        "well": well,
                        "avg_production": avg_prod,
                        "max_production": max_prod,
                        "min_production": min_prod,
                        "std_production": std_prod,
                        "decline_rate": decline_rate if decline_rate != float('inf') else 999,
                        "fluid_ratio": fluid_ratio,
                        "data_points": len(df),
                        "performance_score": performance_score
                    })
                else:
                    failed_wells.append(f"{well} (insufficient data: {len(history) if history else 0} points)")
                    
            except Exception as e:
                logger.warning(f"Error analyzing well {well}: {e}")
                failed_wells.append(f"{well} (analysis error: {str(e)})")
                continue
        
        if not well_performances:
            error_msg = f"Insufficient data for well comparison analysis."
            if failed_wells:
                error_msg += f" Failed wells: {', '.join(failed_wells)}"
            return {
                "type": "error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        
        # ===== SORT AND ANALYZE RESULTS =====
        well_performances.sort(key=lambda x: x["performance_score"], reverse=True)
        
        # ===== CREATE COMPARISON VISUALIZATION =====
        try:
            chart_base64 = self.viz_generator.create_well_comparison_chart(well_performances)
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            chart_base64 = None
        
        # ===== DETAILED ANALYSIS =====
        best_well = well_performances[0]
        worst_well = well_performances[-1]
        
        # Calculate fleet statistics
        fleet_avg_prod = sum([w["avg_production"] for w in well_performances]) / len(well_performances)
        valid_decline_rates = [w["decline_rate"] for w in well_performances if w["decline_rate"] < 999]
        fleet_avg_decline = sum(valid_decline_rates) / len(valid_decline_rates) if valid_decline_rates else 0
        
        # Categorize wells
        high_performers = [w for w in well_performances if w["avg_production"] > fleet_avg_prod * 1.2]
        low_performers = [w for w in well_performances if w["avg_production"] < fleet_avg_prod * 0.8]
        stable_wells = [w for w in well_performances if w["decline_rate"] < fleet_avg_decline]
        
        # Format period info
        start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime('%d %b %Y')
        end_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime('%d %b %Y')
        
        if lang == "id":
            explanation = f"""✅ **Analisis Perbandingan Performa Sumur**

📅 **PERIODE ANALISIS**
• Periode: {start_fmt} - {end_fmt}
• Sumur dianalisis: {len(well_performances)} sumur
• Sumur gagal: {len(failed_wells)} sumur

🏆 **SUMUR TERBAIK: {best_well['well']}**
• Produksi rata-rata: {best_well['avg_production']:.2f} BOPD
• Produksi maksimum: {best_well['max_production']:.2f} BOPD
• Decline rate: {best_well['decline_rate']:.2f}%/tahun
• Variabilitas: {best_well['std_production']:.2f} BOPD
• Rasio fluida: {best_well['fluid_ratio']:.1f}%
• Score performa: {best_well['performance_score']:.1f}

📊 **STATISTIK FLEET**
• Rata-rata produksi fleet: {fleet_avg_prod:.2f} BOPD
• Rata-rata decline rate: {fleet_avg_decline:.2f}%/tahun
• High performers: {len(high_performers)} sumur (>{fleet_avg_prod*1.2:.1f} BOPD)
• Low performers: {len(low_performers)} sumur (<{fleet_avg_prod*0.8:.1f} BOPD)
• Stable wells: {len(stable_wells)} sumur

🎯 **RANKING PERFORMA:**"""
            
            for i, well in enumerate(well_performances[:5]):
                explanation += f"\n{i+1}. {well['well']}: {well['avg_production']:.1f} BOPD (DR: {well['decline_rate']:.1f}%/tahun)"
            
            if len(well_performances) > 5:
                explanation += f"\n... dan {len(well_performances) - 5} sumur lainnya"
            
            explanation += f"""

⚠️ **SUMUR PERLU PERHATIAN: {worst_well['well']}**
• Produksi: {worst_well['avg_production']:.1f} BOPD
• Decline rate: {worst_well['decline_rate']:.1f}%/tahun

🔍 **REKOMENDASI**
• Prioritaskan optimasi pada {len(low_performers)} sumur underperforming
• Pertahankan strategi operasi untuk {len(high_performers)} sumur high-performing  
• Monitor {len(stable_wells)} sumur dengan decline rate stabil"""
            
            if failed_wells:
                explanation += f"\n\n⚠️ **SUMUR TIDAK DAPAT DIANALISIS:**\n• {chr(10).join(failed_wells)}"
                
        else:
            explanation = f"""✅ **Well Performance Comparison Analysis**

📅 **ANALYSIS PERIOD**
• Period: {start_fmt} - {end_fmt}
• Wells analyzed: {len(well_performances)} wells
• Failed wells: {len(failed_wells)} wells

🏆 **BEST PERFORMING WELL: {best_well['well']}**
• Average production: {best_well['avg_production']:.2f} BOPD
• Maximum production: {best_well['max_production']:.2f} BOPD
• Decline rate: {best_well['decline_rate']:.2f}%/year
• Variability: {best_well['std_production']:.2f} BOPD  
• Fluid ratio: {best_well['fluid_ratio']:.1f}%
• Performance score: {best_well['performance_score']:.1f}

📊 **FLEET STATISTICS**
• Fleet average production: {fleet_avg_prod:.2f} BOPD
• Fleet average decline rate: {fleet_avg_decline:.2f}%/year
• High performers: {len(high_performers)} wells (>{fleet_avg_prod*1.2:.1f} BOPD)
• Low performers: {len(low_performers)} wells (<{fleet_avg_prod*0.8:.1f} BOPD)
• Stable wells: {len(stable_wells)} wells

🎯 **PERFORMANCE RANKING:**"""
            
            for i, well in enumerate(well_performances[:5]):
                explanation += f"\n{i+1}. {well['well']}: {well['avg_production']:.1f} BOPD (DR: {well['decline_rate']:.1f}%/year)"
            
            if len(well_performances) > 5:
                explanation += f"\n... and {len(well_performances) - 5} other wells"
            
            explanation += f"""

⚠️ **WELL NEEDS ATTENTION: {worst_well['well']}**
• Production: {worst_well['avg_production']:.1f} BOPD
• Decline rate: {worst_well['decline_rate']:.1f}%/year

🔍 **RECOMMENDATIONS**
• Prioritize optimization for {len(low_performers)} underperforming wells
• Maintain operation strategy for {len(high_performers)} high-performing wells
• Monitor {len(stable_wells)} wells with stable decline rates"""
            
            if failed_wells:
                explanation += f"\n\n⚠️ **WELLS THAT COULD NOT BE ANALYZED:**\n• {chr(10).join(failed_wells)}"
        
        # Generate URL for best well
        interactive_url = self.url_generator.generate_url("history", {"well": best_well['well']})
        
        return {
            "type": "response",
            "explanation": explanation,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat(),
            "comparison_results": {
                "wells_analyzed": len(well_performances),
                "wells_failed": len(failed_wells),
                "period": f"{start_date} to {end_date}",
                "best_well": best_well['well'],
                "fleet_average": fleet_avg_prod,
                "specific_wells_requested": specific_wells
            }
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