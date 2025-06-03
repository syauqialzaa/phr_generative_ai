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

class VisualizationGenerator:
    @staticmethod
    def create_production_chart(data: List[Dict], title: str = "Production History") -> str:
        """Create production chart for /get_history endpoint"""
        try:
            df = pd.DataFrame(data)
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            dates = pd.to_datetime(df['Date'])
            
            # Plot Oil Production
            ax1.plot(dates, df['Production'], 'g-', label='Oil (BOPD)', 
                    linewidth=2, marker='o', markersize=4)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Oil Production (BOPD)', color='g', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='g')
            ax1.grid(True, alpha=0.3)
            
            # Plot Fluid if available
            if 'Fluid' in df.columns:
                ax2 = ax1.twinx()
                ax2.plot(dates, df['Fluid'], 'b-', label='Fluid (BOPD)', 
                        linewidth=2, marker='s', markersize=4)
                ax2.set_ylabel('Fluid Production (BOPD)', color='b', fontsize=12)
                ax2.tick_params(axis='y', labelcolor='b')
            
            # Highlight Job Codes
            if 'JobCode' in df.columns:
                job_points = df[df['JobCode'].notna() & (df['JobCode'] != '')]
                if not job_points.empty:
                    job_dates = pd.to_datetime(job_points['Date'])
                    ax1.scatter(job_dates, job_points['Production'], 
                              color='orange', s=100, marker='^', 
                              label='Job Code', zorder=5)
            
            # Format dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Statistics box
            stats_text = f"Period: {dates.min().strftime('%b %Y')} - {dates.max().strftime('%b %Y')}\n"
            stats_text += f"Avg Oil: {df['Production'].mean():.1f} BOPD\n"
            stats_text += f"Max Oil: {df['Production'].max():.1f} BOPD\n"
            stats_text += f"Min Oil: {df['Production'].min():.1f} BOPD\n"
            stats_text += f"Data Points: {len(df)}"
            
            if 'Fluid' in df.columns:
                stats_text += f"\nAvg Fluid: {df['Fluid'].mean():.1f} BOPD"
            
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Legend
            ax1.legend(loc='upper right')
            if 'Fluid' in df.columns:
                ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9))
            
            fig.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating production chart: {e}")
            return None

    @staticmethod
    def create_dca_analysis_chart(dca_response: Dict) -> str:
        """Create DCA analysis chart for /automatic_dca endpoint"""
        try:
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Extract actual data
            actual_data = dca_response.get('ActualData', [])
            if not actual_data:
                return None
                
            actual_df = pd.DataFrame(actual_data)
            actual_dates = pd.to_datetime(actual_df['date'])
            actual_values = actual_df['value'].values
            
            # Plot actual data
            ax.scatter(actual_dates, actual_values, color='red', s=60, alpha=0.8, 
                      label='Actual Data', zorder=5)
            ax.plot(actual_dates, actual_values, 'r-', alpha=0.6, linewidth=2)
            
            # Generate decline curves from parameters
            decline_rates = dca_response.get('DeclineRate', {})
            exp_params = dca_response.get('Exponential', [])
            harm_params = dca_response.get('Harmonic', [])
            hyper_params = dca_response.get('Hyperbolic', [])
            
            # Create time series for projections (extend 1 year beyond actual data)
            start_date = actual_dates.min()
            end_date = actual_dates.max() + timedelta(days=365)
            time_days = np.arange(0, (end_date - start_date).days + 1)
            projection_dates = [start_date + timedelta(days=int(t)) for t in time_days]
            
            # Define decline models
            def exponential_decline(t, qi, d):
                return qi * np.exp(-d * t)
            
            def harmonic_decline(t, qi, b):
                return qi / (1 + b * t)
            
            def hyperbolic_decline(t, qi, b, n):
                return qi * (1 + b * t) ** (-1 / n)
            
            # Plot decline curves
            colors = {'Exponential': 'purple', 'Harmonic': 'green', 'Hyperbolic': 'blue'}
            
            if exp_params and len(exp_params) >= 2:
                qi, d = exp_params[0], exp_params[1]
                exp_values = [exponential_decline(t, qi, d) for t in time_days]
                ax.plot(projection_dates, exp_values, '--', color=colors['Exponential'], 
                       linewidth=2, alpha=0.8, 
                       label=f'Exponential Decline (DR: {decline_rates.get("Exponential", "N/A")}%/year)')
            
            if harm_params and len(harm_params) >= 2:
                qi, b = harm_params[0], harm_params[1]
                harm_values = [harmonic_decline(t, qi, b) for t in time_days]
                ax.plot(projection_dates, harm_values, '--', color=colors['Harmonic'], 
                       linewidth=2, alpha=0.8,
                       label=f'Harmonic Decline (DR: {decline_rates.get("Harmonic", "N/A")}%/year)')
            
            if hyper_params and len(hyper_params) >= 3:
                qi, b, n = hyper_params[0], hyper_params[1], hyper_params[2]
                hyper_values = [hyperbolic_decline(t, qi, b, n) for t in time_days]
                ax.plot(projection_dates, hyper_values, '--', color=colors['Hyperbolic'], 
                       linewidth=2, alpha=0.8,
                       label=f'Hyperbolic Decline (DR: {decline_rates.get("Hyperbolic", "N/A")}%/year)')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Production (BOPD)', fontsize=12)
            ax.set_title('DCA Analysis with Decline Curves', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add analysis summary
            summary_text = "DCA Analysis Summary:\n"
            for model, rate in decline_rates.items():
                summary_text += f"{model}: {rate}%/year\n"
            
            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            fig.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating DCA analysis chart: {e}")
            return None

    @staticmethod
    def create_production_prediction_chart(prediction_response: Dict, well_name: str = "") -> str:
        """Create production prediction chart for /predict_production endpoint"""
        try:
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Extract predictions
            exp_pred = prediction_response.get('ExponentialPrediction', [])
            harm_pred = prediction_response.get('HarmonicPrediction', [])
            hyper_pred = prediction_response.get('HyperbolicPrediction', [])
            
            if not any([exp_pred, harm_pred, hyper_pred]):
                return None
            
            # Plot predictions
            colors = {'Exponential': 'purple', 'Harmonic': 'green', 'Hyperbolic': 'blue'}
            
            if exp_pred:
                dates = pd.to_datetime([p['date'] for p in exp_pred])
                values = [p['value'] for p in exp_pred]
                ax.plot(dates, values, '-', color=colors['Exponential'], 
                       linewidth=3, alpha=0.8, label='Exponential Prediction')
                
                # Add markers for key points
                ax.scatter(dates[0], values[0], color=colors['Exponential'], 
                          s=100, marker='o', zorder=5)
                ax.scatter(dates[-1], values[-1], color=colors['Exponential'], 
                          s=100, marker='s', zorder=5)
            
            if harm_pred:
                dates = pd.to_datetime([p['date'] for p in harm_pred])
                values = [p['value'] for p in harm_pred]
                ax.plot(dates, values, '--', color=colors['Harmonic'], 
                       linewidth=3, alpha=0.8, label='Harmonic Prediction')
            
            if hyper_pred:
                dates = pd.to_datetime([p['date'] for p in hyper_pred])
                values = [p['value'] for p in hyper_pred]
                ax.plot(dates, values, '-.', color=colors['Hyperbolic'], 
                       linewidth=3, alpha=0.8, label='Hyperbolic Prediction')
            
            # Add economic limit line if we can infer it
            if exp_pred:
                economic_limit = exp_pred[-1]['value']
                ax.axhline(y=economic_limit, color='red', linestyle=':', 
                          linewidth=2, alpha=0.7, label=f'Economic Limit ({economic_limit:.1f} BOPD)')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Production (BOPD)', fontsize=12)
            title = f'Production Prediction - {well_name}' if well_name else 'Production Prediction'
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add prediction summary
            if exp_pred:
                start_date = exp_pred[0]['date']
                end_date = exp_pred[-1]['date']
                start_prod = exp_pred[0]['value']
                end_prod = exp_pred[-1]['value']
                duration_days = len(exp_pred)
                
                summary_text = f"Prediction Summary:\n"
                summary_text += f"Start: {start_date} ({start_prod:.1f} BOPD)\n"
                summary_text += f"End: {end_date} ({end_prod:.1f} BOPD)\n"
                summary_text += f"Duration: {duration_days} days\n"
                summary_text += f"Total Decline: {((start_prod - end_prod)/start_prod*100):.1f}%"
                
                props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9)
                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
            
            fig.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating prediction chart: {e}")
            return None

    @staticmethod
    def create_ml_prediction_chart(ml_response: Dict) -> str:
        """Create ML prediction chart for /predict_ml endpoint"""
        try:
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Extract ML data
            dates_actual = ml_response.get('dates_actual', [])
            actual = ml_response.get('actual', [])
            predicted = ml_response.get('predicted', [])
            dates_extended = ml_response.get('dates_extended', [])
            extended_prediction = ml_response.get('extended_prediction', [])
            elr_threshold = ml_response.get('elr_threshold', 10.0)
            
            if not dates_actual or not actual:
                return None
            
            # Convert dates
            actual_dates = pd.to_datetime(dates_actual)
            
            # Plot actual vs predicted (historical)
            ax.scatter(actual_dates, actual, color='blue', s=30, alpha=0.6, 
                      label='Actual Production', zorder=3)
            
            if predicted:
                ax.plot(actual_dates, predicted, 'orange', linewidth=2, alpha=0.8,
                       label='ML Model (Historical)', zorder=4)
            
            # Plot future predictions
            if dates_extended and extended_prediction:
                extended_dates = pd.to_datetime(dates_extended)
                ax.plot(extended_dates, extended_prediction, 'green', linewidth=3, 
                       alpha=0.8, label='ML Prediction (Future)', zorder=4)
                
                # Add markers for start and end of prediction
                ax.scatter(extended_dates[0], extended_prediction[0], 
                          color='green', s=100, marker='o', zorder=5)
                ax.scatter(extended_dates[-1], extended_prediction[-1], 
                          color='green', s=100, marker='s', zorder=5)
            
            # Add ELR threshold line
            ax.axhline(y=elr_threshold, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'ELR Threshold ({elr_threshold} BOPD)')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Production (BOPD)', fontsize=12)
            ax.set_title('Machine Learning Production Prediction', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add ML summary
            if actual and predicted:
                mse = np.mean([(a - p)**2 for a, p in zip(actual, predicted)])
                rmse = np.sqrt(mse)
                
                summary_text = f"ML Model Performance:\n"
                summary_text += f"RMSE: {rmse:.2f}\n"
                summary_text += f"Historical Points: {len(actual)}\n"
                
                if extended_prediction:
                    summary_text += f"Future Points: {len(extended_prediction)}\n"
                    summary_text += f"Final Prediction: {extended_prediction[-1]:.1f} BOPD"
                
                props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.9)
                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
            
            fig.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating ML prediction chart: {e}")
            return None

    @staticmethod
    def create_well_comparison_chart(well_performances: List[Dict]) -> str:
        """Create well comparison chart"""
        try:
            # Sort by production
            sorted_wells = sorted(well_performances, key=lambda x: x['avg_production'], reverse=True)
            
            wells = [w['well'] for w in sorted_wells[:10]]  # Top 10 wells
            productions = [w['avg_production'] for w in sorted_wells[:10]]
            decline_rates = [w['decline_rate'] for w in sorted_wells[:10]]
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Bar chart for production
            bars = ax1.bar(range(len(wells)), productions, color='skyblue', 
                          alpha=0.7, label='Avg Production (BOPD)')
            ax1.set_xlabel('Wells', fontsize=12)
            ax1.set_ylabel('Average Production (BOPD)', color='blue', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xticks(range(len(wells)))
            ax1.set_xticklabels(wells, rotation=45, ha='right')
            
            # Line chart for decline rate
            ax2 = ax1.twinx()
            ax2.plot(range(len(wells)), decline_rates, 'ro-', linewidth=2, 
                    markersize=6, label='Decline Rate (%/yr)')
            ax2.set_ylabel('Decline Rate (%/year)', color='red', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(productions)*0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            
            plt.title('Well Performance Comparison', fontsize=16, fontweight='bold', pad=20)
            
            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            fig.tight_layout()
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return None