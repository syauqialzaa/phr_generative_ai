import base64
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, List
import logging

matplotlib.use('Agg')
logger = logging.getLogger(__name__)

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
                        name=f'{model_name} (DR: {decline_rates.get(model_name, "N/A")}%/yr)',
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
    
    @staticmethod
    def create_well_comparison_chart(well_performances: List[Dict]) -> str:
        """Create well comparison chart"""
        try:
            # Sort by production
            sorted_wells = sorted(well_performances, key=lambda x: x['avg_production'], reverse=True)
            
            wells = [w['well'] for w in sorted_wells[:10]]  # Top 10 wells
            productions = [w['avg_production'] for w in sorted_wells[:10]]
            decline_rates = [w['decline_rate'] for w in sorted_wells[:10]]
            
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Bar chart for production
            bars = ax1.bar(range(len(wells)), productions, color='skyblue', alpha=0.7, label='Avg Production (BOPD)')
            ax1.set_xlabel('Wells')
            ax1.set_ylabel('Average Production (BOPD)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xticks(range(len(wells)))
            ax1.set_xticklabels(wells, rotation=45, ha='right')
            
            # Line chart for decline rate
            ax2 = ax1.twinx()
            ax2.plot(range(len(wells)), decline_rates, 'ro-', linewidth=2, markersize=6, label='Decline Rate (%/yr)')
            ax2.set_ylabel('Decline Rate (%/year)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(productions)*0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
            
            plt.title('Well Performance Comparison', fontsize=14, fontweight='bold')
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