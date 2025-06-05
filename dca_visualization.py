import base64
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta

matplotlib.use('Agg')
logger = logging.getLogger(__name__)

class DCAVisualizationGenerator:
    @staticmethod
    def create_production_chart(data: List[Dict], title: str = "Production History") -> str:
        """Create production chart for /get_history endpoint with enhanced data handling"""
        try:
            if not data or len(data) == 0:
                logger.warning("No data provided for production chart")
                return None
                
            df = pd.DataFrame(data)
            
            # Validate required columns
            if 'Date' not in df.columns or 'Production' not in df.columns:
                logger.error(f"Missing required columns. Available: {list(df.columns)}")
                return None
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Convert and validate dates
            try:
                dates = pd.to_datetime(df['Date'])
                if dates.isna().any():
                    logger.warning("Some dates could not be parsed")
                    # Remove rows with invalid dates
                    valid_mask = ~dates.isna()
                    dates = dates[valid_mask]
                    df = df[valid_mask]
            except Exception as e:
                logger.error(f"Date conversion failed: {e}")
                return None
            
            # Plot Oil Production with enhanced styling
            production_values = pd.to_numeric(df['Production'], errors='coerce')
            if production_values.isna().all():
                logger.error("No valid production data found")
                return None
            
            ax1.plot(dates, production_values, 'g-', label='Oil Production (BOPD)', 
                    linewidth=2.5, marker='o', markersize=5, alpha=0.8)
            ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Oil Production (BOPD)', color='g', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='g')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Plot Fluid if available
            if 'Fluid' in df.columns:
                fluid_values = pd.to_numeric(df['Fluid'], errors='coerce')
                if not fluid_values.isna().all():
                    ax2 = ax1.twinx()
                    ax2.plot(dates, fluid_values, 'b-', label='Fluid Production (BFPD)', 
                            linewidth=2.5, marker='s', markersize=5, alpha=0.8)
                    ax2.set_ylabel('Fluid Production (BFPD)', color='b', fontsize=12, fontweight='bold')
                    ax2.tick_params(axis='y', labelcolor='b')
            
            # Highlight Job Codes with enhanced markers
            if 'JobCode' in df.columns:
                job_mask = df['JobCode'].notna() & (df['JobCode'] != '') & (df['JobCode'] != 'None')
                job_points = df[job_mask]
                if not job_points.empty:
                    job_dates = pd.to_datetime(job_points['Date'])
                    job_production = pd.to_numeric(job_points['Production'], errors='coerce')
                    
                    ax1.scatter(job_dates, job_production, 
                              color='orange', s=120, marker='^', 
                              label=f'Job Codes ({len(job_points)})', zorder=5,
                              edgecolors='red', linewidth=1.5)
                    
                    # Add annotations for job codes
                    for i, (date, prod, job) in enumerate(zip(job_dates, job_production, job_points['JobCode'])):
                        if i < 5:  # Limit annotations to avoid clutter
                            ax1.annotate(str(job), (date, prod), 
                                       xytext=(10, 10), textcoords='offset points',
                                       fontsize=8, alpha=0.8,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Enhanced date formatting
            date_range = (dates.max() - dates.min()).days
            if date_range > 730:  # > 2 years
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            elif date_range > 365:  # > 1 year
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            else:
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Enhanced title
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Enhanced statistics box
            stats_text = f"Period: {dates.min().strftime('%d %b %Y')} - {dates.max().strftime('%d %b %Y')}\n"
            stats_text += f"Avg Oil: {production_values.mean():.1f} BOPD\n"
            stats_text += f"Max Oil: {production_values.max():.1f} BOPD\n"
            stats_text += f"Min Oil: {production_values.min():.1f} BOPD\n"
            stats_text += f"Data Points: {len(df)}\n"
            
            # Calculate decline trend
            if len(production_values) > 1:
                trend = ((production_values.iloc[-1] - production_values.iloc[0]) / production_values.iloc[0]) * 100
                trend_text = f"Trend: {trend:+.1f}%"
                stats_text += trend_text
            
            if 'Fluid' in df.columns and not fluid_values.isna().all():
                stats_text += f"\nAvg Fluid: {fluid_values.mean():.1f} BFPD"
                # Water cut calculation
                water_cut = ((fluid_values.mean() - production_values.mean()) / fluid_values.mean()) * 100
                if water_cut >= 0:
                    stats_text += f"\nWater Cut: {water_cut:.1f}%"
            
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='navy')
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, fontweight='bold')
            
            # Enhanced legend
            ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
            if 'Fluid' in df.columns and not fluid_values.isna().all():
                ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9), fontsize=10, framealpha=0.9)
            
            fig.tight_layout()
            
            # Save with high quality
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info(f"Successfully created production chart with {len(df)} data points")
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating production chart: {e}")
            return None

    @staticmethod
    def create_dca_analysis_chart(dca_response: Dict) -> str:
        """Create DCA analysis chart with enhanced visualization based on actual API response"""
        try:
            if not dca_response or 'ActualData' not in dca_response:
                logger.error("Invalid DCA response: missing ActualData")
                return None
                
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Extract and validate actual data
            actual_data = dca_response.get('ActualData', [])
            if not actual_data:
                logger.error("No actual data found in DCA response")
                return None
                
            try:
                actual_df = pd.DataFrame(actual_data)
                
                # Handle different date field names
                date_field = None
                for field in ['date', 'Date', 'DATE']:
                    if field in actual_df.columns:
                        date_field = field
                        break
                
                if not date_field:
                    logger.error(f"No date field found. Available columns: {list(actual_df.columns)}")
                    return None
                
                # Handle different value field names
                value_field = None
                for field in ['value', 'production', 'Production', 'rate']:
                    if field in actual_df.columns:
                        value_field = field
                        break
                
                if not value_field:
                    logger.error(f"No value field found. Available columns: {list(actual_df.columns)}")
                    return None
                
                actual_dates = pd.to_datetime(actual_df[date_field])
                actual_values = pd.to_numeric(actual_df[value_field], errors='coerce')
                
                # Remove invalid data points
                valid_mask = ~(actual_dates.isna() | actual_values.isna())
                actual_dates = actual_dates[valid_mask]
                actual_values = actual_values[valid_mask]
                
                if len(actual_dates) == 0:
                    logger.error("No valid actual data points after cleaning")
                    return None
                
            except Exception as e:
                logger.error(f"Error processing actual data: {e}")
                return None
            
            # Plot actual data with enhanced styling
            ax.scatter(actual_dates, actual_values, color='red', s=80, alpha=0.8, 
                      label=f'Actual Data ({len(actual_values)} points)', zorder=5, edgecolors='darkred')
            ax.plot(actual_dates, actual_values, 'r-', alpha=0.6, linewidth=2)
            
            # Extract decline parameters and rates
            decline_rates = dca_response.get('DeclineRate', {})
            exp_params = dca_response.get('Exponential', [])
            harm_params = dca_response.get('Harmonic', [])
            hyper_params = dca_response.get('Hyperbolic', [])
            
            # Create extended time series for projections
            start_date = actual_dates.min()
            end_date = actual_dates.max() + timedelta(days=365)  # Extend 1 year
            time_days = np.arange(0, (end_date - start_date).days + 1)
            projection_dates = [start_date + timedelta(days=int(t)) for t in time_days]
            
            # Define decline models with proper error handling
            def safe_exponential_decline(t, qi, d):
                try:
                    return qi * np.exp(-d * t / 365)  # Convert daily decline
                except:
                    return np.full_like(t, qi)
            
            def safe_harmonic_decline(t, qi, d):
                try:
                    return qi / (1 + d * t / 365)
                except:
                    return np.full_like(t, qi)
            
            def safe_hyperbolic_decline(t, qi, d, b):
                try:
                    if b <= 0 or b >= 1:
                        b = 0.5  # Default b-factor
                    return qi / ((1 + b * d * t / 365) ** (1/b))
                except:
                    return np.full_like(t, qi)
            
            # Plot decline curves with validation
            colors = {'Exponential': 'purple', 'Harmonic': 'green', 'Hyperbolic': 'blue'}
            
            curves_plotted = 0
            
            # Exponential decline
            if exp_params and len(exp_params) >= 2:
                try:
                    qi, d = float(exp_params[0]), float(exp_params[1])
                    if qi > 0 and d > 0:
                        exp_values = safe_exponential_decline(time_days, qi, d)
                        ax.plot(projection_dates, exp_values, '--', color=colors['Exponential'], 
                               linewidth=3, alpha=0.8, 
                               label=f'Exponential (Qi:{qi:.1f}, DR:{decline_rates.get("Exponential", "N/A")}%/yr)')
                        curves_plotted += 1
                except Exception as e:
                    logger.warning(f"Error plotting exponential curve: {e}")
            
            # Harmonic decline
            if harm_params and len(harm_params) >= 2:
                try:
                    qi, d = float(harm_params[0]), float(harm_params[1])
                    if qi > 0 and d > 0:
                        harm_values = safe_harmonic_decline(time_days, qi, d)
                        ax.plot(projection_dates, harm_values, '--', color=colors['Harmonic'], 
                               linewidth=3, alpha=0.8,
                               label=f'Harmonic (Qi:{qi:.1f}, DR:{decline_rates.get("Harmonic", "N/A")}%/yr)')
                        curves_plotted += 1
                except Exception as e:
                    logger.warning(f"Error plotting harmonic curve: {e}")
            
            # Hyperbolic decline
            if hyper_params and len(hyper_params) >= 3:
                try:
                    qi, d, b = float(hyper_params[0]), float(hyper_params[1]), float(hyper_params[2])
                    if qi > 0 and d > 0 and 0 < b < 1:
                        hyper_values = safe_hyperbolic_decline(time_days, qi, d, b)
                        ax.plot(projection_dates, hyper_values, '--', color=colors['Hyperbolic'], 
                               linewidth=3, alpha=0.8,
                               label=f'Hyperbolic (Qi:{qi:.1f}, b:{b:.2f}, DR:{decline_rates.get("Hyperbolic", "N/A")}%/yr)')
                        curves_plotted += 1
                except Exception as e:
                    logger.warning(f"Error plotting hyperbolic curve: {e}")
            
            # Enhanced chart formatting
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Production (BOPD)', fontsize=12, fontweight='bold')
            ax.set_title('DCA Analysis with Decline Curves', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            # Format dates based on data range
            date_range = (actual_dates.max() - actual_dates.min()).days
            if date_range > 730:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            else:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add R-squared values if available
            r_squared_data = dca_response.get('RSquared', {})
            analysis_summary = "DCA Analysis Summary:\n"
            
            for model, rate in decline_rates.items():
                r2_value = r_squared_data.get(model, 0)
                analysis_summary += f"{model}: DR={rate:.1f}%/yr, R²={r2_value:.3f}\n"
            
            analysis_summary += f"\nData Points: {len(actual_values)}"
            analysis_summary += f"\nCurves Plotted: {curves_plotted}"
            
            # Add best model recommendation
            if r_squared_data:
                best_model = max(r_squared_data, key=r_squared_data.get)
                best_r2 = r_squared_data[best_model]
                analysis_summary += f"\nBest Model: {best_model} (R²={best_r2:.3f})"
            
            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
            ax.text(0.02, 0.98, analysis_summary, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props, fontweight='bold')
            
            fig.tight_layout()
            
            # Save with high quality
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info(f"Successfully created DCA analysis chart with {curves_plotted} curves")
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating DCA analysis chart: {e}")
            return None

    @staticmethod
    def create_production_prediction_chart(prediction_response: Dict, well_name: str = "") -> str:
        """Create production prediction chart with enhanced visualization"""
        try:
            if not prediction_response:
                logger.error("Empty prediction response")
                return None
                
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Extract predictions with validation
            exp_pred = prediction_response.get('ExponentialPrediction', [])
            harm_pred = prediction_response.get('HarmonicPrediction', [])
            hyper_pred = prediction_response.get('HyperbolicPrediction', [])
            
            if not any([exp_pred, harm_pred, hyper_pred]):
                logger.error("No prediction data found in response")
                return None
            
            # Colors for different models
            colors = {'Exponential': 'purple', 'Harmonic': 'green', 'Hyperbolic': 'blue'}
            predictions_plotted = 0
            economic_limit = None
            
            # Plot Exponential prediction
            if exp_pred:
                try:
                    dates = pd.to_datetime([p['date'] for p in exp_pred])
                    values = [p['value'] for p in exp_pred]
                    
                    ax.plot(dates, values, '-', color=colors['Exponential'], 
                           linewidth=3, alpha=0.8, label=f'Exponential Prediction ({len(values)} days)')
                    
                    # Add markers for key points
                    ax.scatter(dates[0], values[0], color=colors['Exponential'], 
                              s=120, marker='o', zorder=5, edgecolors='black', linewidth=1)
                    ax.scatter(dates[-1], values[-1], color=colors['Exponential'], 
                              s=120, marker='s', zorder=5, edgecolors='black', linewidth=1)
                    
                    # Annotate start and end points
                    ax.annotate(f'Start: {values[0]:.1f} BOPD', (dates[0], values[0]),
                               xytext=(10, 10), textcoords='offset points', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpurple', alpha=0.8))
                    ax.annotate(f'ELR: {values[-1]:.1f} BOPD', (dates[-1], values[-1]),
                               xytext=(10, -15), textcoords='offset points', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpurple', alpha=0.8))
                    
                    economic_limit = values[-1]
                    predictions_plotted += 1
                    
                except Exception as e:
                    logger.warning(f"Error plotting exponential prediction: {e}")
            
            # Plot Harmonic prediction
            if harm_pred:
                try:
                    dates = pd.to_datetime([p['date'] for p in harm_pred])
                    values = [p['value'] for p in harm_pred]
                    
                    ax.plot(dates, values, '--', color=colors['Harmonic'], 
                           linewidth=3, alpha=0.8, label=f'Harmonic Prediction ({len(values)} days)')
                    predictions_plotted += 1
                    
                except Exception as e:
                    logger.warning(f"Error plotting harmonic prediction: {e}")
            
            # Plot Hyperbolic prediction
            if hyper_pred:
                try:
                    dates = pd.to_datetime([p['date'] for p in hyper_pred])
                    values = [p['value'] for p in hyper_pred]
                    
                    ax.plot(dates, values, '-.', color=colors['Hyperbolic'], 
                           linewidth=3, alpha=0.8, label=f'Hyperbolic Prediction ({len(values)} days)')
                    predictions_plotted += 1
                    
                except Exception as e:
                    logger.warning(f"Error plotting hyperbolic prediction: {e}")
            
            # Add economic limit line
            if economic_limit is not None:
                ax.axhline(y=economic_limit, color='red', linestyle=':', 
                          linewidth=2, alpha=0.7, label=f'Economic Limit ({economic_limit:.1f} BOPD)')
            
            # Enhanced chart formatting
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Production (BOPD)', fontsize=12, fontweight='bold')
            title = f'Production Prediction - {well_name}' if well_name else 'Production Prediction'
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            # Format dates
            if exp_pred:
                start_date = pd.to_datetime(exp_pred[0]['date'])
                end_date = pd.to_datetime(exp_pred[-1]['date'])
                date_range = (end_date - start_date).days
                
                if date_range > 365:
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                else:
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                    
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add prediction summary
            if exp_pred:
                start_date_str = exp_pred[0]['date']
                end_date_str = exp_pred[-1]['date']
                start_prod = exp_pred[0]['value']
                end_prod = exp_pred[-1]['value']
                duration_days = len(exp_pred)
                total_production = sum([p['value'] for p in exp_pred])
                
                summary_text = f"Prediction Summary:\n"
                summary_text += f"Period: {start_date_str} to {end_date_str}\n"
                summary_text += f"Duration: {duration_days} days ({duration_days/30:.1f} months)\n"
                summary_text += f"Start: {start_prod:.1f} BOPD → End: {end_prod:.1f} BOPD\n"
                summary_text += f"Total Production: {total_production:,.0f} barrels\n"
                summary_text += f"Total Decline: {((start_prod - end_prod)/start_prod*100):.1f}%\n"
                summary_text += f"Models Plotted: {predictions_plotted}"
                
                props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen')
                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props, fontweight='bold')
            
            fig.tight_layout()
            
            # Save with high quality
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info(f"Successfully created prediction chart with {predictions_plotted} models")
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating prediction chart: {e}")
            return None

    @staticmethod
    def create_ml_prediction_chart(ml_response: Dict) -> str:
        """Create ML prediction chart with enhanced visualization and confidence intervals"""
        try:
            if not ml_response:
                logger.error("Empty ML response")
                return None
                
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Extract ML data with validation
            dates_actual = ml_response.get('dates_actual', [])
            actual = ml_response.get('actual', [])
            predicted = ml_response.get('predicted', [])
            dates_extended = ml_response.get('dates_extended', [])
            extended_prediction = ml_response.get('extended_prediction', [])
            elr_threshold = ml_response.get('elr_threshold', 10.0)
            confidence_interval = ml_response.get('confidence_interval', {})
            
            if not dates_actual or not actual:
                logger.error("No actual data found in ML response")
                return None
            
            # Convert and validate dates
            try:
                actual_dates = pd.to_datetime(dates_actual)
                actual_values = pd.to_numeric(actual, errors='coerce')
                
                # Remove invalid data
                valid_mask = ~(actual_dates.isna() | actual_values.isna())
                actual_dates = actual_dates[valid_mask]
                actual_values = actual_values[valid_mask]
                
                if len(actual_dates) == 0:
                    logger.error("No valid actual data after cleaning")
                    return None
                    
            except Exception as e:
                logger.error(f"Error processing actual data: {e}")
                return None
            
            # Plot actual vs predicted (historical)
            ax.scatter(actual_dates, actual_values, color='blue', s=50, alpha=0.7, 
                      label=f'Actual Production ({len(actual_values)} points)', zorder=3)
            
            if predicted and len(predicted) == len(actual_values):
                predicted_values = pd.to_numeric(predicted, errors='coerce')
                if not predicted_values.isna().all():
                    ax.plot(actual_dates, predicted_values, 'orange', linewidth=2.5, alpha=0.8,
                           label='ML Model (Historical Fit)', zorder=4)
                    
                    # Calculate and display R²
                    try:
                        correlation_matrix = np.corrcoef(actual_values, predicted_values)
                        r_squared = correlation_matrix[0, 1] ** 2
                        logger.info(f"ML Model R² = {r_squared:.3f}")
                    except:
                        r_squared = None
            
            # Plot future predictions
            if dates_extended and extended_prediction:
                try:
                    extended_dates = pd.to_datetime(dates_extended)
                    extended_values = pd.to_numeric(extended_prediction, errors='coerce')
                    
                    # Remove invalid future data
                    valid_future_mask = ~(extended_dates.isna() | extended_values.isna())
                    extended_dates = extended_dates[valid_future_mask]
                    extended_values = extended_values[valid_future_mask]
                    
                    if len(extended_dates) > 0:
                        ax.plot(extended_dates, extended_values, 'green', linewidth=3, 
                               alpha=0.8, label=f'ML Prediction (Future - {len(extended_values)} points)', zorder=4)
                        
                        # Add markers for start and end of prediction
                        ax.scatter(extended_dates.iloc[0], extended_values.iloc[0], 
                                  color='green', s=120, marker='o', zorder=5, edgecolors='darkgreen')
                        ax.scatter(extended_dates.iloc[-1], extended_values.iloc[-1], 
                                  color='green', s=120, marker='s', zorder=5, edgecolors='darkgreen')
                        
                        # Annotate key points
                        ax.annotate(f'Future Start: {extended_values.iloc[0]:.1f} BOPD', 
                                   (extended_dates.iloc[0], extended_values.iloc[0]),
                                   xytext=(10, 10), textcoords='offset points', fontsize=9,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
                        
                except Exception as e:
                    logger.warning(f"Error plotting future predictions: {e}")
            
            # Plot confidence intervals if available
            if confidence_interval and dates_extended:
                try:
                    upper_bound = confidence_interval.get('upper', [])
                    lower_bound = confidence_interval.get('lower', [])
                    
                    if upper_bound and lower_bound and len(upper_bound) == len(extended_dates):
                        upper_values = pd.to_numeric(upper_bound, errors='coerce')
                        lower_values = pd.to_numeric(lower_bound, errors='coerce')
                        
                        ax.fill_between(extended_dates, lower_values, upper_values, 
                                       alpha=0.3, color='green', label='95% Confidence Interval')
                        
                except Exception as e:
                    logger.warning(f"Error plotting confidence intervals: {e}")
            
            # Add ELR threshold line
            ax.axhline(y=elr_threshold, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'ELR Threshold ({elr_threshold} BOPD)')
            
            # Enhanced chart formatting
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Production (BOPD)', fontsize=12, fontweight='bold')
            ax.set_title('Machine Learning Production Prediction', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            # Format dates based on data range
            all_dates = list(actual_dates)
            if dates_extended:
                all_dates.extend(pd.to_datetime(dates_extended))
            
            if len(all_dates) > 0:
                date_range = (max(all_dates) - min(all_dates)).days
                if date_range > 730:
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                else:
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add comprehensive ML summary
            if actual and predicted:
                try:
                    mse = np.mean([(a - p)**2 for a, p in zip(actual, predicted)])
                    rmse = np.sqrt(mse)
                    mae = np.mean([abs(a - p) for a, p in zip(actual, predicted)])
                    
                    summary_text = f"ML Model Performance:\n"
                    summary_text += f"RMSE: {rmse:.2f} BOPD\n"
                    summary_text += f"MAE: {mae:.2f} BOPD\n"
                    
                    if 'r_squared' in locals():
                        summary_text += f"R²: {r_squared:.3f}\n"
                    
                    summary_text += f"Historical Points: {len(actual)}\n"
                    
                    if extended_prediction:
                        summary_text += f"Future Points: {len(extended_prediction)}\n"
                        summary_text += f"Final Prediction: {extended_prediction[-1]:.1f} BOPD\n"
                        summary_text += f"Days to ELR: {len(extended_prediction)}"
                    
                    if confidence_interval:
                        summary_text += f"\nConfidence Intervals: Available"
                    
                    props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='teal')
                    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=props, fontweight='bold')
                           
                except Exception as e:
                    logger.warning(f"Error calculating ML performance metrics: {e}")
            
            fig.tight_layout()
            
            # Save with high quality
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info("Successfully created ML prediction chart")
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating ML prediction chart: {e}")
            return None

    @staticmethod
    def create_well_comparison_chart(well_performances: List[Dict]) -> str:
        """Create enhanced well comparison chart with multiple metrics"""
        try:
            if not well_performances:
                logger.error("No well performance data provided")
                return None
                
            # Sort by performance score and limit to top wells
            sorted_wells = sorted(well_performances, key=lambda x: x['avg_production'], reverse=True)
            top_wells = sorted_wells[:10]  # Top 10 wells for visibility
            
            wells = [w['well'] for w in top_wells]
            productions = [w['avg_production'] for w in top_wells]
            decline_rates = [w['decline_rate'] if w['decline_rate'] < 999 else 0 for w in top_wells]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # Top subplot: Production comparison
            bars = ax1.bar(range(len(wells)), productions, color='skyblue', 
                          alpha=0.7, edgecolor='navy', linewidth=1)
            ax1.set_ylabel('Average Production (BOPD)', fontsize=12, fontweight='bold')
            ax1.set_title('Well Production Comparison', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(productions)*0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Color code bars based on performance
            fleet_avg = sum(productions) / len(productions)
            for i, bar in enumerate(bars):
                if productions[i] > fleet_avg * 1.2:
                    bar.set_color('green')
                    bar.set_alpha(0.8)
                elif productions[i] < fleet_avg * 0.8:
                    bar.set_color('red')
                    bar.set_alpha(0.8)
            
            # Bottom subplot: Decline rate comparison
            line = ax2.plot(range(len(wells)), decline_rates, 'ro-', linewidth=2.5, 
                           markersize=8, alpha=0.8)
            ax2.set_xlabel('Wells', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Decline Rate (%/year)', color='red', fontsize=12, fontweight='bold')
            ax2.set_title('Well Decline Rate Comparison', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(True, alpha=0.3)
            
            # Set x-axis labels for both subplots
            for ax in [ax1, ax2]:
                ax.set_xticks(range(len(wells)))
                ax.set_xticklabels(wells, rotation=45, ha='right')
            
            # Add performance categories legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.8, label='High Performer (>120% avg)'),
                Patch(facecolor='skyblue', alpha=0.7, label='Average Performer'),
                Patch(facecolor='red', alpha=0.8, label='Low Performer (<80% avg)')
            ]
            ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Add summary statistics
            summary_text = f"Fleet Statistics:\n"
            summary_text += f"Wells Analyzed: {len(well_performances)}\n"
            summary_text += f"Avg Production: {fleet_avg:.1f} BOPD\n"
            summary_text += f"Best Well: {wells[0]} ({productions[0]:.1f} BOPD)\n"
            summary_text += f"Range: {min(productions):.1f} - {max(productions):.1f} BOPD"
            
            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
            ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, fontweight='bold')
            
            plt.tight_layout()
            
            # Save with high quality
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info(f"Successfully created well comparison chart for {len(wells)} wells")
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return None