import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from io import BytesIO
from typing import Dict, List, Any, Optional
import logging
import os
from PIL import Image
import requests
import aiohttp

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

class WellboreVisualizationGenerator:
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self.img_cache = {}  # Cache for downloaded images
        
        # Icon mapping for wellbore components (same as app.js)
        self.icon_mapping = {
            # Casing components
            'SurfCsg': 'SurfCsg.png',
            'SurfCsg2': 'SurfCsg2.png', 
            'IntermCsg': 'IntermCsg.png',
            'ProdCsg': 'ProdCsg.png',
            'ProdCsgUp': 'ProdCsgUp.png',
            'ProdCsgSwedge': 'ProdCsgSwedge.png',
            'CsgSwedge': 'CsgSwedge.png',
            'ScgSwedgeUp': 'ScgSwedgeUp.png',
            
            # Tubing components
            'Tubing': 'Tubing.png',
            'TubingUp': 'TubingUp.png',
            'TubingSI': 'TubingSI.png',
            'TubingCblUp': 'TubingCblUp.png',
            'TubungCbl': 'TubungCbl.png',
            
            # ESP System
            'ESPump': 'ESPump.png',
            'ESPump_New': 'ESPump_New.png',
            'ESPumpCbl': 'ESPumpCbl.png',
            'Motor': 'Motor.png',
            'PIntake': 'PIntake.png',
            'Seal': 'Seal.png',
            'Protector': 'Protector.png',
            'GasSeparator': 'GasSeparator.png',
            
            # Completion components
            'Perfo1': 'Perfo1.png',
            'PerfoOpen': 'PerfoOpen.png',
            'PerfoCls': 'PerfoCls.png',
            'PerfoJoint': 'PerfoJoint.png',
            'PerfoSqz': 'PerfoSqz.png',
            'PerfoSqz1': 'PerfoSqz1.png',
            
            # Packers and plugs
            'PKR': 'PKR.png',
            'TbgPacker': 'TbgPacker.png',
            'BPlug': 'BPlug.png',
            'BPlug_Tbg': 'BPlug_Tbg.png',
            
            # Other components
            'PBTD': 'PBTD.png',
            'PBTD2': 'PBTD2.png',
            'OpenHole': 'OpenHole.png',
            'Fill': 'Fill.png',
            'Fish': 'Fish.png',
            'TOC': 'TOC.png',
            'TOC_SI': 'TOC_SI.png',
            'TopSand': 'TopSand.png',
            'TBA': 'TBA.png',
            'Venturi': 'Venturi.png',
            'SlottedJoint': 'SlottedJoint.png',
            'TbgSLJoint': 'TbgSLJoint.png',
            'TbgPump': 'TbgPump.png',
            'ProdLnr': 'ProdLnr.png',
            
            # Control and monitoring
            'CPD': 'CPD.png',
            'CPFD': 'CPFD.png',
            'CPFU': 'CPFU.png',
            'ECP': 'ECP.png',
            
            # Geometric elements
            'DepthLn': 'DepthLn.png',
            'Line 51': 'Line 51.png',
            'Line 52': 'Line 52.png',
            'AutoShape 46': 'AutoShape 46.png',
            'AutoShape 47': 'AutoShape 47.png',
            'Rectangle 14': 'Rectangle 14.png',
            'Rectangle 49': 'Rectangle 49.png',
            'Rectangle 50': 'Rectangle 50.png',
            'Rectangle 56': 'Rectangle 56.png',
            'Rectangle 57': 'Rectangle 57.png'
        }
    
    async def download_image(self, image_filename: str) -> Optional[np.ndarray]:
        """Download image from API server and cache it"""
        if image_filename in self.img_cache:
            return self.img_cache[image_filename]
        
        try:
            # Try local file first
            local_path = os.path.join('img', image_filename)
            if os.path.exists(local_path):
                img = Image.open(local_path)
                # Convert to RGBA if not already
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                img_array = np.array(img)
                self.img_cache[image_filename] = img_array
                return img_array
            
            # Try API download
            img_url = f"{self.api_base_url}/img/{image_filename}"
            async with aiohttp.ClientSession() as session:
                async with session.get(img_url) as response:
                    if response.status == 200:
                        img_data = await response.read()
                        img = Image.open(BytesIO(img_data))
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        img_array = np.array(img)
                        self.img_cache[image_filename] = img_array
                        return img_array
                    else:
                        logger.warning(f"Could not download image: {image_filename}")
                        return None
        except Exception as e:
            logger.error(f"Error downloading image {image_filename}: {e}")
            return None
    
    def get_icon_filename(self, icon_name: str) -> str:
        """Get the actual filename for an icon"""
        # Direct mapping
        if icon_name in self.icon_mapping:
            return self.icon_mapping[icon_name]
        
        # Try case-insensitive search
        icon_lower = icon_name.lower()
        for key, filename in self.icon_mapping.items():
            if key.lower() == icon_lower:
                return filename
        
        # Try partial matching
        for key, filename in self.icon_mapping.items():
            if icon_lower in key.lower() or key.lower() in icon_lower:
                return filename
        
        # Default fallback
        logger.warning(f"Icon not found: {icon_name}, using default")
        return 'Rectangle 14.png'
    
    async def create_wellbore_diagram(self, wellbore_data: List[Dict], well_name: str, title: str = None) -> str:
        """Create wellbore diagram using positioning logic from app.js"""
        try:
            if not wellbore_data:
                return self._create_no_data_chart()
            
            # Set up the plot - mimic app.js dimensions and styling
            fig, ax = plt.subplots(figsize=(14, 20))
            fig.patch.set_facecolor('white')
            
            # Sort data by BOT_MD like in app.js
            data = sorted(wellbore_data, key=lambda x: x.get('BOT_MD', 0))
            
            # Calculate dimensions like app.js
            max_depth = max([comp.get('BOT_MD', 0) for comp in data])
            max_od = max([comp.get('OD_INCH', 1) for comp in data])
            
            # Set plot limits - similar to app.js SVG dimensions
            ax.set_xlim(-250, 250)  # Centered around 0, like app.js width=450 centered
            ax.set_ylim(max_depth + 50, -50)  # Inverted Y-axis, depth down
            
            # Draw depth scale (like app.js scaleGroup)
            self._draw_depth_scale(ax, max_depth)
            
            # Draw main wellbore centerline
            ax.axvline(x=0, color='black', linewidth=2, alpha=0.8, linestyle='-', zorder=1)
            
            # Separate components by type (following app.js logic)
            perforation_open = [d for d in data if d.get('ICON_NAME') == 'PerfoOpen']
            perforation_close = [d for d in data if d.get('ICON_NAME') == 'PerfoCls']
            perforation_sqz = [d for d in data if d.get('ICON_NAME') == 'PerfoSqz']
            tbg_pump = [d for d in data if d.get('ICON_NAME') == 'TbgPump']
            tubing = [d for d in data if d.get('ICON_NAME') == 'Tubing']
            other_components = [d for d in data if d.get('ICON_NAME') not in ['TbgPump', 'Tubing']]
            
            # Sort each group by BOT_MD (like app.js)
            other_components.sort(key=lambda x: x.get('BOT_MD', 0))
            tubing.sort(key=lambda x: x.get('BOT_MD', 0))
            perforation_open.sort(key=lambda x: x.get('BOT_MD', 0))
            perforation_close.sort(key=lambda x: x.get('BOT_MD', 0))
            perforation_sqz.sort(key=lambda x: x.get('BOT_MD', 0))
            
            # Render in the same order as app.js
            sorted_components = (other_components + tubing + tbg_pump + 
                               perforation_open + perforation_close + perforation_sqz)
            
            # Track remark positions like app.js
            remark_positions = {'left': [], 'right': []}
            
            # Process each component using app.js logic
            for index, component in enumerate(sorted_components):
                await self._render_component_js_style(ax, component, index, max_depth, max_od, remark_positions)
            
            # Customize the plot
            ax.set_xlabel('Horizontal Distance', fontsize=12, fontweight='bold')
            ax.set_ylabel('Measured Depth (ft)', fontsize=12, fontweight='bold')
            
            if title:
                ax.set_title(f'{title}\nWell: {well_name}', fontsize=16, fontweight='bold', pad=20)
            else:
                ax.set_title(f'Depth Analysis (0-TD ft)\nWell: {well_name}', fontsize=16, fontweight='bold', pad=20)
            
            # Add component legend
            self._add_component_legend_js_style(ax, data)
            
            # Add statistics box like app.js
            self._add_statistics_box(ax, data)
            
            # Turn off x-axis ticks like app.js
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating wellbore diagram: {e}")
            return self._create_error_chart(str(e))
    
    def _draw_depth_scale(self, ax, max_depth: float):
        """Draw depth scale like app.js scaleGroup"""
        # Draw scale lines every 200 ft (like app.js)
        for depth in range(0, int(max_depth) + 200, 200):
            # Draw tick mark at left side (like app.js x1=60, x2=80)
            ax.plot([-240, -220], [depth, depth], 'k-', linewidth=1, alpha=0.7)
            
            # Add depth label
            ax.text(-250, depth, str(depth), fontsize=10, ha='right', va='center')
    
    async def _render_component_js_style(self, ax, component: Dict, index: int, max_depth: float, max_od: float, remark_positions: Dict):
        """Render component using app.js positioning logic"""
        try:
            icon_name = component.get('ICON_NAME', 'Rectangle 14')
            top_md = component.get('TOP_MD', 0)
            bot_md = component.get('BOT_MD', 0)
            od_inch = component.get('OD_INCH', 1.0)
            remarks = component.get('REMARKS', '')
            
            # Validate component dimensions (like app.js)
            if top_md >= bot_md:
                logger.warning(f"Invalid component dimensions: {component}")
                return
            
            # Calculate dimensions using app.js logic
            min_height = 10  # MIN_HEIGHT from app.js
            img_height = max(min_height, bot_md - top_md)
            
            # Width scaling like app.js: widthScale = od => (od / maxOD) * 150
            img_width = (od_inch / max_od) * 150
            
            # Position calculation like app.js: xPos = (width / 2) - (imgWidth / 2) + 200
            # app.js width = 450, so center = 225, but we use 0 as center in matplotlib
            x_pos = -(img_width / 2)  # Center the component
            y_pos = top_md
            
            # Get the appropriate image filename
            image_filename = self.get_icon_filename(icon_name)
            
            # Download and cache the image
            img_array = await self.download_image(image_filename)
            
            if img_array is not None:
                # Create image object with proper sizing (like app.js preserveAspectRatio="none")
                zoom_factor = min(img_width / img_array.shape[1], img_height / img_array.shape[0]) * 0.5
                
                # Create image object
                imagebox = OffsetImage(img_array, zoom=zoom_factor)
                ab = AnnotationBbox(imagebox, (x_pos + img_width/2, y_pos + img_height/2), 
                                  frameon=False, pad=0, zorder=5)
                ax.add_artist(ab)
            else:
                # Fallback rectangle
                rect = patches.Rectangle((x_pos, y_pos), img_width, img_height,
                                       linewidth=1, edgecolor='black', 
                                       facecolor=self._get_component_color(icon_name), 
                                       alpha=0.7, zorder=3)
                ax.add_patch(rect)
            
            # Add remarks if available (using app.js logic)
            if remarks:
                self._add_remarks_js_style(ax, component, index, x_pos, img_width, remark_positions)
                
        except Exception as e:
            logger.warning(f"Error rendering component {component.get('ICON_NAME', 'Unknown')}: {e}")
    
    def _add_remarks_js_style(self, ax, component: Dict, index: int, x_pos: float, img_width: float, remark_positions: Dict):
        """Add remarks using app.js positioning logic"""
        icon_name = component.get('ICON_NAME', '')
        remarks = component.get('REMARKS', '')
        bot_md = component.get('BOT_MD', 0)
        
        # Determine side like app.js: isLeft = index % 2 === 0
        is_left = index % 2 == 0
        line_length = 40  # Like app.js
        
        # Calculate y position with collision avoidance (like app.js)
        y_bottom = bot_md
        side = "left" if is_left else "right"
        
        # Collision detection (like app.js remarkPositions logic)
        for pos in remark_positions[side]:
            if abs(y_bottom - pos) < 40:
                y_bottom += 40
        remark_positions[side].append(y_bottom)
        
        # Draw line (like app.js line element)
        line_x1 = x_pos - 10 if is_left else x_pos + img_width + 10
        line_x2 = x_pos - line_length if is_left else x_pos + img_width + line_length
        
        ax.plot([line_x1, line_x2], [bot_md, y_bottom], 'r-', linewidth=2, zorder=4)
        
        # Add text label (like app.js text element)
        text_x = x_pos - line_length - 5 if is_left else x_pos + img_width + line_length + 5
        ha = 'right' if is_left else 'left'
        
        label_text = f"{icon_name}: {remarks}"
        ax.text(text_x, y_bottom, label_text, fontsize=10, ha=ha, va='center',
               color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
               zorder=6)
    
    def _get_component_color(self, icon_name: str) -> str:
        """Get color for component based on type"""
        icon_lower = icon_name.lower()
        
        if any(term in icon_lower for term in ['csg', 'casing']):
            return 'lightblue'
        elif any(term in icon_lower for term in ['tubing', 'tbg']):
            return 'lightgreen'
        elif any(term in icon_lower for term in ['pump', 'motor', 'esp']):
            return 'orange'
        elif any(term in icon_lower for term in ['perfo', 'perf']):
            return 'red'
        elif any(term in icon_lower for term in ['pkr', 'packer']):
            return 'yellow'
        else:
            return 'lightcoral'
    
    def _add_component_legend_js_style(self, ax, components: List[Dict]):
        """Add a legend like app.js component summary"""
        component_types = {}
        for comp in components:
            icon_name = comp.get('ICON_NAME', 'Unknown')
            comp_type = self._categorize_component(icon_name)
            if comp_type not in component_types:
                component_types[comp_type] = 0
            component_types[comp_type] += 1
        
        legend_text = "Component Summary:\n"
        for comp_type, count in sorted(component_types.items()):
            legend_text += f"• {comp_type}: {count}\n"
        
        # Position like app.js (bottom right)
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='bottom', horizontalalignment='right', 
               bbox=props, fontweight='bold')
    
    def _add_statistics_box(self, ax, components: List[Dict]):
        """Add statistics box like app.js"""
        total_components = len(components)
        min_depth = min([comp.get('TOP_MD', 0) for comp in components])
        max_depth = max([comp.get('BOT_MD', 0) for comp in components])
        depth_range = max_depth - min_depth
        
        stats_text = f"Components: {total_components}\n"
        stats_text += f"Depth Range: {min_depth:,.0f} - {max_depth:,.0f} ft\n"
        stats_text += f"Total Interval: {depth_range:,.0f} ft"
        
        # Position like app.js (top left)
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props, fontweight='bold')
    
    def _categorize_component(self, icon_name: str) -> str:
        """Categorize component based on icon name"""
        icon_lower = icon_name.lower()
        
        if any(term in icon_lower for term in ['surfcsg', 'surface']):
            return 'Surface Casing'
        elif any(term in icon_lower for term in ['intermcsg', 'interm']):
            return 'Intermediate Casing'
        elif any(term in icon_lower for term in ['prodcsg', 'production']):
            return 'Production Casing'
        elif any(term in icon_lower for term in ['tubing', 'tbg']):
            return 'Tubing'
        elif any(term in icon_lower for term in ['pump', 'esp']):
            return 'ESP Pump'
        elif 'motor' in icon_lower:
            return 'ESP Motor'
        elif any(term in icon_lower for term in ['perfo', 'perf']):
            return 'Perforations'
        elif any(term in icon_lower for term in ['pkr', 'packer']):
            return 'Packers'
        elif 'seal' in icon_lower:
            return 'Seals'
        else:
            return 'Other Components'
    
    def _create_no_data_chart(self) -> str:
        """Create a chart when no data is available"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No wellbore data available', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_error_chart(self, error_message: str) -> str:
        """Create an error chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Error creating wellbore diagram:\n{error_message}', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=14, color='red', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def generate_wellbore_summary(self, wellbore_data: List[Dict], well_name: str, lang: str = "en") -> str:
        """Generate comprehensive wellbore analysis summary"""
        if not wellbore_data:
            return "No wellbore data available for analysis."
        
        # Analyze components
        total_components = len(wellbore_data)
        min_depth = min([comp.get('TOP_MD', 0) for comp in wellbore_data])
        max_depth = max([comp.get('BOT_MD', 0) for comp in wellbore_data])
        depth_range = max_depth - min_depth
        
        # Categorize components
        component_summary = {}
        esp_components = []
        casing_components = []
        completion_components = []
        
        for comp in wellbore_data:
            icon_name = comp.get('ICON_NAME', '')
            category = self._categorize_component(icon_name)
            
            if category not in component_summary:
                component_summary[category] = 0
            component_summary[category] += 1
            
            if any(term in icon_name.lower() for term in ['pump', 'motor', 'esp', 'seal', 'intake']):
                esp_components.append(comp)
            elif 'csg' in icon_name.lower():
                casing_components.append(comp)
            elif any(term in icon_name.lower() for term in ['perfo', 'tubing', 'pkr']):
                completion_components.append(comp)
        
        if lang == "id":
            summary = f"""Analisis Komprehensif Wellbore - {well_name}

📊 OVERVIEW WELLBORE
• Total komponen: {total_components} items
• Rentang kedalaman: {min_depth:,.0f} - {max_depth:,.0f} ft
• Total interval: {depth_range:,.0f} ft

🏗️ KOMPONEN UTAMA"""
            
            for category, count in component_summary.items():
                summary += f"\n• {category}: {count} unit"
            
            summary += f"\n\n🔧 ANALISIS SISTEM"
            
            if esp_components:
                summary += f"\n• ESP System: {len(esp_components)} komponen terdeteksi"
                summary += f"\n  - Artificial lift system aktif"
                esp_depth_range = f"{min([c.get('TOP_MD', 0) for c in esp_components]):,.0f} - {max([c.get('BOT_MD', 0) for c in esp_components]):,.0f} ft"
                summary += f"\n  - Kedalaman ESP: {esp_depth_range}"
            
            if casing_components:
                summary += f"\n• Casing Program: {len(casing_components)} string"
                casing_sizes = [comp.get('OD_INCH', 0) for comp in casing_components if comp.get('OD_INCH', 0) > 0]
                if casing_sizes:
                    summary += f"\n  - Ukuran: {max(casing_sizes):.1f}\" hingga {min(casing_sizes):.1f}\""
            
            if completion_components:
                summary += f"\n• Completion: {len(completion_components)} komponen"
                if any('perfo' in comp.get('ICON_NAME', '').lower() for comp in completion_components):
                    perfo_count = sum(1 for comp in completion_components if 'perfo' in comp.get('ICON_NAME', '').lower())
                    summary += f"\n  - Zona perforasi: {perfo_count} interval"
            
            summary += f"\n\n💡 REKOMENDASI"
            if esp_components:
                summary += f"\n• Monitor performa ESP secara berkala"
                summary += f"\n• Surveillance sistem artificial lift"
            summary += f"\n• Inspeksi integritas wellbore rutin"
            summary += f"\n• Evaluasi kondisi completion equipment"
            
        else:
            summary = f"""Comprehensive Wellbore Analysis - {well_name}

📊 WELLBORE OVERVIEW
• Total components: {total_components} items
• Depth range: {min_depth:,.0f} - {max_depth:,.0f} ft
• Total interval: {depth_range:,.0f} ft

🏗️ MAJOR COMPONENTS"""
            
            for category, count in component_summary.items():
                summary += f"\n• {category}: {count} units"
            
            summary += f"\n\n🔧 SYSTEM ANALYSIS"
            
            if esp_components:
                summary += f"\n• ESP System: {len(esp_components)} components detected"
                summary += f"\n  - Active artificial lift system"
                esp_depth_range = f"{min([c.get('TOP_MD', 0) for c in esp_components]):,.0f} - {max([c.get('BOT_MD', 0) for c in esp_components]):,.0f} ft"
                summary += f"\n  - ESP depth: {esp_depth_range}"
            
            if casing_components:
                summary += f"\n• Casing Program: {len(casing_components)} strings"
                casing_sizes = [comp.get('OD_INCH', 0) for comp in casing_components if comp.get('OD_INCH', 0) > 0]
                if casing_sizes:
                    summary += f"\n  - Sizes: {max(casing_sizes):.1f}\" to {min(casing_sizes):.1f}\""
            
            if completion_components:
                summary += f"\n• Completion: {len(completion_components)} components"
                if any('perfo' in comp.get('ICON_NAME', '').lower() for comp in completion_components):
                    perfo_count = sum(1 for comp in completion_components if 'perfo' in comp.get('ICON_NAME', '').lower())
                    summary += f"\n  - Perforation zones: {perfo_count} intervals"
            
            summary += f"\n\n💡 RECOMMENDATIONS"
            if esp_components:
                summary += f"\n• Monitor ESP performance regularly"
                summary += f"\n• Conduct artificial lift surveillance"
            summary += f"\n• Routine wellbore integrity inspection"
            summary += f"\n• Evaluate completion equipment condition"
        
        return summary