import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

from wellbore_client import WellboreApiClient
from wellbore_query_processor import WellboreQueryProcessor
from wellbore_visualization import WellboreVisualizationGenerator
from wellbore_url_generator import WellboreUrlGenerator
from vector_store import VectorStoreManager
from config import WELLBORE_API_BASE_URL, WELLBORE_APP_BASE_URL, MILVUS_DB_PATH

logger = logging.getLogger(__name__)

class WellboreAssistant:
    def __init__(self):
        self.query_processor = WellboreQueryProcessor()
        self.viz_generator = WellboreVisualizationGenerator(WELLBORE_API_BASE_URL)
        self.url_generator = WellboreUrlGenerator()
        self.vector_store = VectorStoreManager(MILVUS_DB_PATH)
    
    async def process_query(self, query: str, history: List[Dict] = None) -> Dict:
        """Process wellbore-related query and return comprehensive response"""
        
        # Detect language
        detected_lang = self.query_processor.detect_language(query)
        
        # Check if query is wellbore-related
        if not self.query_processor.is_wellbore_related(query):
            return {
                "type": "response",
                "explanation": self.query_processor.get_non_wellbore_response(detected_lang),
                "timestamp": datetime.now().isoformat()
            }
        
        # Detect intent and extract parameters
        intent = self.query_processor.detect_wellbore_intent(query)
        params = self.query_processor.extract_wellbore_parameters(query)
        
        # Process based on intent
        try:
            async with WellboreApiClient(WELLBORE_API_BASE_URL) as client:
                if intent == "show_diagram":
                    return await self._handle_show_diagram(client, params, detected_lang)
                elif intent == "wellbore_analysis":
                    return await self._handle_wellbore_analysis(client, params, detected_lang)
                elif intent == "casing_analysis":
                    return await self._handle_casing_analysis(client, params, detected_lang)
                elif intent == "completion_analysis":
                    return await self._handle_completion_analysis(client, params, detected_lang)
                elif intent == "depth_analysis":
                    return await self._handle_depth_analysis(client, params, detected_lang)
                else:
                    return await self._handle_general_wellbore(query, detected_lang)
                    
        except Exception as e:
            logger.error(f"Error processing wellbore query: {e}")
            return {
                "type": "error",
                "message": self.query_processor.get_error_message(detected_lang, str(e)),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _handle_show_diagram(self, client: WellboreApiClient, params: Dict, lang: str) -> Dict:
        """Handle wellbore diagram visualization requests"""
        # Get wellbore data
        wellbore_data = await client.get_wellbore_data(
            uwi=params.get("uwi"),
            top_md=params.get("top_md"),
            bot_md=params.get("bot_md"),
            icon_name=params.get("icon_name")
        )
        
        if not wellbore_data:
            return {
                "type": "response",
                "explanation": "No wellbore data found for the specified parameters." if lang == "en" 
                             else "Tidak ditemukan data wellbore untuk parameter yang ditentukan.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate wellbore diagram
        chart_base64 = await self.viz_generator.create_wellbore_diagram(
            wellbore_data, params.get("uwi", "Unknown")
        )
        
        # Generate comprehensive analysis
        analysis = self.viz_generator.generate_wellbore_summary(
            wellbore_data, params.get("uwi", "Unknown"), lang
        )
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("diagram", params)
        
        return {
            "type": "response",
            "explanation": analysis,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_wellbore_analysis(self, client: WellboreApiClient, params: Dict, lang: str) -> Dict:
        """Handle comprehensive wellbore analysis"""
        # Get wellbore data
        wellbore_data = await client.get_wellbore_data(
            uwi=params.get("uwi"),
            top_md=params.get("top_md"),
            bot_md=params.get("bot_md")
        )
        
        if not wellbore_data:
            return {
                "type": "response",
                "explanation": "No wellbore data available for analysis." if lang == "en"
                             else "Tidak ada data wellbore yang tersedia untuk analisis.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate diagram and analysis
        chart_base64 = await self.viz_generator.create_wellbore_diagram(
            wellbore_data, params.get("uwi", "Unknown")
        )
        
        # Enhanced analysis with detailed insights
        analysis = self.viz_generator.generate_wellbore_summary(
            wellbore_data, params.get("uwi", "Unknown"), lang
        )
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("analysis", params)
        
        return {
            "type": "response",
            "explanation": analysis,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_casing_analysis(self, client: WellboreApiClient, params: Dict, lang: str) -> Dict:
        """Handle casing-specific analysis"""
        # Filter for casing components
        params["icon_name"] = "csg"
        
        wellbore_data = await client.get_wellbore_data(
            uwi=params.get("uwi"),
            icon_name="csg"
        )
        
        if not wellbore_data:
            return {
                "type": "response",
                "explanation": "No casing data found for the specified well." if lang == "en"
                             else "Tidak ditemukan data casing untuk sumur yang ditentukan.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate focused casing analysis
        chart_base64 = await self.viz_generator.create_wellbore_diagram(
            wellbore_data, params.get("uwi", "Unknown"), "Casing Configuration Analysis"
        )
        
        # Casing-specific summary
        analysis = self._generate_casing_summary(wellbore_data, params.get("uwi", "Unknown"), lang)
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("casing", params)
        
        return {
            "type": "response",
            "explanation": analysis,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_completion_analysis(self, client: WellboreApiClient, params: Dict, lang: str) -> Dict:
        """Handle completion-specific analysis"""
        # Get all wellbore data to analyze completion
        wellbore_data = await client.get_wellbore_data(
            uwi=params.get("uwi")
        )
        
        if not wellbore_data:
            return {
                "type": "response",
                "explanation": "No completion data found for the specified well." if lang == "en"
                             else "Tidak ditemukan data completion untuk sumur yang ditentukan.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Filter completion-related components
        completion_components = [comp for comp in wellbore_data 
                               if any(keyword in comp.get('ICON_NAME', '').lower() 
                                     for keyword in ['tubing', 'perfo', 'esp', 'pump', 'motor', 'pkr', 'seal'])]
        
        if not completion_components:
            return {
                "type": "response",
                "explanation": "No completion components found in wellbore data." if lang == "en"
                             else "Tidak ditemukan komponen completion dalam data wellbore.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate completion analysis
        chart_base64 = await self.viz_generator.create_wellbore_diagram(
            completion_components, params.get("uwi", "Unknown"), "Completion Analysis"
        )
        
        # Completion-specific summary
        analysis = self._generate_completion_summary(completion_components, params.get("uwi", "Unknown"), lang)
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("completion", params)
        
        return {
            "type": "response",
            "explanation": analysis,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_depth_analysis(self, client: WellboreApiClient, params: Dict, lang: str) -> Dict:
        """Handle depth-specific analysis"""
        wellbore_data = await client.get_wellbore_data(
            uwi=params.get("uwi"),
            top_md=params.get("top_md"),
            bot_md=params.get("bot_md")
        )
        
        if not wellbore_data:
            return {
                "type": "response",
                "explanation": "No data found for the specified depth range." if lang == "en"
                             else "Tidak ditemukan data untuk rentang kedalaman yang ditentukan.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate depth-focused analysis
        chart_base64 = await self.viz_generator.create_wellbore_diagram(
            wellbore_data, params.get("uwi", "Unknown"), 
            f"Depth Analysis ({params.get('top_md', 0)}-{params.get('bot_md', 'TD')} ft)"
        )
        
        # Depth-specific summary
        analysis = self._generate_depth_summary(wellbore_data, params, lang)
        
        # Generate interactive URL
        interactive_url = self.url_generator.generate_url("diagram", params)
        
        return {
            "type": "response",
            "explanation": analysis,
            "visualization": chart_base64,
            "app_url": interactive_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_general_wellbore(self, query: str, lang: str) -> Dict:
        """Handle general wellbore questions"""
        if lang == "id":
            explanation = """Wellbore Diagram & Analysis - Panduan Komprehensif

🏗️ KOMPONEN WELLBORE UTAMA
• **Surface Casing**: Casing terluar untuk isolasi formasi dangkal
• **Intermediate Casing**: Casing tengah untuk isolasi zona masalah  
• **Production Casing**: Casing produksi untuk zona produktif
• **Tubing**: Pipa produksi di dalam casing
• **Completion Equipment**: Peralatan komplesi (packer, perforasi, dll)

⚡ ARTIFICIAL LIFT SYSTEMS
• **ESP (Electric Submersible Pump)**: Sistem pompa listrik bawah permukaan
• **Motor**: Motor listrik penggerak ESP
• **Pump Intake**: Saluran masuk pompa
• **Seal Section**: Bagian penyekat motor

🎯 COMPLETION COMPONENTS  
• **Perforations**: Lubang perforasi untuk aliran fluida
• **Packers**: Penyekat annulus
• **Safety Valves**: Katup pengaman
• **Flow Control Devices**: Perangkat kontrol aliran

🔍 ANALISIS YANG TERSEDIA
• Visualisasi diagram wellbore lengkap dengan komponen
• Analisis konfigurasi casing dan ukuran
• Evaluasi sistem completion dan artificial lift
• Identifikasi komponen kritis dan rekomendasi
• Analisis kedalaman dan distribusi komponen

💡 KEGUNAAN PRAKTIS
• Perencanaan intervensi sumur
• Troubleshooting masalah produksi  
• Optimasi completion design
• Risk assessment wellbore integrity
• Training dan dokumentasi

Sistem wellbore terintegrasi memerlukan analisis menyeluruh untuk memastikan performa optimal dan keamanan operasi."""
        else:
            explanation = """Wellbore Diagram & Analysis - Comprehensive Guide

🏗️ MAJOR WELLBORE COMPONENTS
• **Surface Casing**: Outermost casing for shallow formation isolation
• **Intermediate Casing**: Middle casing for problematic zone isolation
• **Production Casing**: Production casing for productive zones
• **Tubing**: Production tubing inside casing
• **Completion Equipment**: Completion tools (packers, perforations, etc.)

⚡ ARTIFICIAL LIFT SYSTEMS
• **ESP (Electric Submersible Pump)**: Downhole electric pump system
• **Motor**: Electric motor driving ESP
• **Pump Intake**: Pump inlet section
• **Seal Section**: Motor sealing component

🎯 COMPLETION COMPONENTS
• **Perforations**: Perforation holes for fluid flow
• **Packers**: Annulus sealers
• **Safety Valves**: Safety shutdown valves
• **Flow Control Devices**: Flow control equipment

🔍 AVAILABLE ANALYSES
• Complete wellbore diagram visualization with components
• Casing configuration and sizing analysis
• Completion and artificial lift system evaluation
• Critical component identification and recommendations
• Depth analysis and component distribution

💡 PRACTICAL APPLICATIONS
• Well intervention planning
• Production troubleshooting
• Completion design optimization
• Wellbore integrity risk assessment
• Training and documentation

Integrated wellbore systems require comprehensive analysis to ensure optimal performance and operational safety."""
        
        return {
            "type": "response",
            "explanation": explanation,
            "app_url": WELLBORE_APP_BASE_URL,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_casing_summary(self, casing_data: List[Dict], uwi: str, lang: str) -> str:
        """Generate casing-specific analysis summary"""
        if not casing_data:
            return "No casing data available."
        
        # Analyze casing strings
        casing_strings = []
        for casing in casing_data:
            icon_name = casing.get('ICON_NAME', '')
            if 'csg' in icon_name.lower():
                casing_strings.append({
                    'type': icon_name,
                    'top': casing.get('TOP_MD', 0),
                    'bottom': casing.get('BOT_MD', 0),
                    'size': casing.get('OD_INCH', 0),
                    'remarks': casing.get('REMARKS', '')
                })
        
        # Sort by size (largest to smallest)
        casing_strings.sort(key=lambda x: x['size'], reverse=True)
        
        if lang == "id":
            summary = f"""Analisis Konfigurasi Casing - {uwi}

🔧 PROGRAM CASING
• Total casing strings: {len(casing_strings)}
• Desain: {"Multi-stage" if len(casing_strings) > 2 else "Conventional"}

📐 DETAIL CASING STRINGS"""
            
            for i, casing in enumerate(casing_strings):
                stage_name = ["Surface", "Intermediate", "Production"][min(i, 2)]
                summary += f"\n• {stage_name} Casing: {casing['size']:.1f}\" @ {casing['bottom']:,} ft"
                if casing['remarks']:
                    summary += f"\n  Spec: {casing['remarks']}"
            
            # Design analysis
            if len(casing_strings) >= 3:
                summary += f"\n\n🎯 EVALUASI DESAIN"
                summary += f"\n• Telescoping design dengan {len(casing_strings)} stages"
                summary += f"\n• Depth capability: {max([c['bottom'] for c in casing_strings]):,} ft"
                summary += f"\n• Size reduction: {casing_strings[0]['size']:.1f}\" → {casing_strings[-1]['size']:.1f}\""
            
            summary += f"\n\n💡 REKOMENDASI"
            summary += f"\n• Monitor cement integrity pada semua casing strings"
            summary += f"\n• Lakukan pressure test berkala"
            summary += f"\n• Evaluasi corrosion protection program"
        else:
            summary = f"""Casing Configuration Analysis - {uwi}

🔧 CASING PROGRAM
• Total casing strings: {len(casing_strings)}
• Design: {"Multi-stage" if len(casing_strings) > 2 else "Conventional"}

📐 CASING STRING DETAILS"""
            
            for i, casing in enumerate(casing_strings):
                stage_name = ["Surface", "Intermediate", "Production"][min(i, 2)]
                summary += f"\n• {stage_name} Casing: {casing['size']:.1f}\" @ {casing['bottom']:,} ft"
                if casing['remarks']:
                    summary += f"\n  Spec: {casing['remarks']}"
            
            # Design analysis
            if len(casing_strings) >= 3:
                summary += f"\n\n🎯 DESIGN EVALUATION"
                summary += f"\n• Telescoping design with {len(casing_strings)} stages"
                summary += f"\n• Depth capability: {max([c['bottom'] for c in casing_strings]):,} ft"
                summary += f"\n• Size reduction: {casing_strings[0]['size']:.1f}\" → {casing_strings[-1]['size']:.1f}\""
            
            summary += f"\n\n💡 RECOMMENDATIONS"
            summary += f"\n• Monitor cement integrity across all casing strings"
            summary += f"\n• Conduct regular pressure testing"
            summary += f"\n• Evaluate corrosion protection program"
        
        return summary
    
    def _generate_completion_summary(self, completion_data: List[Dict], uwi: str, lang: str) -> str:
        """Generate completion-specific analysis summary"""
        if not completion_data:
            return "No completion data available."
        
        # Analyze completion components
        esp_components = [c for c in completion_data if 'esp' in c.get('ICON_NAME', '').lower() or 'pump' in c.get('ICON_NAME', '').lower()]
        perforations = [c for c in completion_data if 'perfo' in c.get('ICON_NAME', '').lower()]
        packers = [c for c in completion_data if 'pkr' in c.get('ICON_NAME', '').lower()]
        tubing = [c for c in completion_data if 'tubing' in c.get('ICON_NAME', '').lower()]
        
        if lang == "id":
            summary = f"""Analisis Completion System - {uwi}

🎯 KONFIGURASI COMPLETION
• Tipe: {"ESP Completion" if esp_components else "Natural Flow"}
• Komponen: {len(completion_data)} items

🔧 SISTEM ARTIFICIAL LIFT"""
            if esp_components:
                summary += f"\n• ESP System: {len(esp_components)} komponen"
                for esp in esp_components:
                    summary += f"\n  - {esp.get('ICON_NAME', '')}: {esp.get('TOP_MD', 0)}-{esp.get('BOT_MD', 0)} ft"
            else:
                summary += f"\n• Natural flow completion"
            
            summary += f"\n\n🎯 COMPLETION DETAILS"
            if perforations:
                summary += f"\n• Perforasi: {len(perforations)} zona"
                for perf in perforations:
                    summary += f"\n  - {perf.get('TOP_MD', 0)}-{perf.get('BOT_MD', 0)} ft"
            
            if packers:
                summary += f"\n• Packers: {len(packers)} unit"
            
            if tubing:
                summary += f"\n• Tubing: {len(tubing)} string"
            
            summary += f"\n\n💡 EVALUASI OPERASIONAL"
            if esp_components:
                summary += f"\n• ESP monitoring dan surveillance diperlukan"
                summary += f"\n• Predictive maintenance untuk ESP components"
            summary += f"\n• Regular completion integrity assessment"
            summary += f"\n• Production optimization review"
        else:
            summary = f"""Completion System Analysis - {uwi}

🎯 COMPLETION CONFIGURATION
• Type: {"ESP Completion" if esp_components else "Natural Flow"}
• Components: {len(completion_data)} items

🔧 ARTIFICIAL LIFT SYSTEM"""
            if esp_components:
                summary += f"\n• ESP System: {len(esp_components)} components"
                for esp in esp_components:
                    summary += f"\n  - {esp.get('ICON_NAME', '')}: {esp.get('TOP_MD', 0)}-{esp.get('BOT_MD', 0)} ft"
            else:
                summary += f"\n• Natural flow completion"
            
            summary += f"\n\n🎯 COMPLETION DETAILS"
            if perforations:
                summary += f"\n• Perforations: {len(perforations)} zones"
                for perf in perforations:
                    summary += f"\n  - {perf.get('TOP_MD', 0)}-{perf.get('BOT_MD', 0)} ft"
            
            if packers:
                summary += f"\n• Packers: {len(packers)} units"
            
            if tubing:
                summary += f"\n• Tubing: {len(tubing)} strings"
            
            summary += f"\n\n💡 OPERATIONAL EVALUATION"
            if esp_components:
                summary += f"\n• ESP monitoring and surveillance required"
                summary += f"\n• Predictive maintenance for ESP components"
            summary += f"\n• Regular completion integrity assessment"
            summary += f"\n• Production optimization review"
        
        return summary
    
    def _generate_depth_summary(self, depth_data: List[Dict], params: Dict, lang: str) -> str:
        """Generate depth-specific analysis summary"""
        if not depth_data:
            return "No data available for specified depth range."
        
        top_md = params.get('top_md', 0)
        bot_md = params.get('bot_md', max([d.get('BOT_MD', 0) for d in depth_data]))
        
        # Analyze components by depth
        components_by_type = {}
        for comp in depth_data:
            icon_name = comp.get('ICON_NAME', '')
            if icon_name not in components_by_type:
                components_by_type[icon_name] = []
            components_by_type[icon_name].append(comp)
        
        if lang == "id":
            summary = f"""Analisis Kedalaman {top_md:,} - {bot_md:,} ft

📏 RENTANG ANALISIS
• Kedalaman target: {top_md:,} - {bot_md:,} ft
• Total komponen: {len(depth_data)} items
• Interval: {bot_md - top_md:,} ft

🔧 DISTRIBUSI KOMPONEN"""
            
            for comp_type, comps in components_by_type.items():
                summary += f"\n• {comp_type}: {len(comps)} unit"
                depth_range = f"{min([c.get('TOP_MD', 0) for c in comps]):,} - {max([c.get('BOT_MD', 0) for c in comps]):,} ft"
                summary += f" ({depth_range})"
            
            summary += f"\n\n🎯 KARAKTERISTIK ZONA"
            if any('csg' in comp.get('ICON_NAME', '').lower() for comp in depth_data):
                summary += f"\n• Zona casing/conductor"
            if any('perfo' in comp.get('ICON_NAME', '').lower() for comp in depth_data):
                summary += f"\n• Zona produktif dengan perforasi"
            if any('esp' in comp.get('ICON_NAME', '').lower() for comp in depth_data):
                summary += f"\n• Zona artificial lift (ESP)"
            
            summary += f"\n\n💡 REKOMENDASI OPERASIONAL"
            summary += f"\n• Focus monitoring pada zona kritis"
            summary += f"\n• Interval inspection sesuai component type"
            summary += f"\n• Depth-specific maintenance planning"
        else:
            summary = f"""Depth Analysis {top_md:,} - {bot_md:,} ft

📏 ANALYSIS RANGE
• Target depth: {top_md:,} - {bot_md:,} ft
• Total components: {len(depth_data)} items
• Interval: {bot_md - top_md:,} ft

🔧 COMPONENT DISTRIBUTION"""
            
            for comp_type, comps in components_by_type.items():
                summary += f"\n• {comp_type}: {len(comps)} units"
                depth_range = f"{min([c.get('TOP_MD', 0) for c in comps]):,} - {max([c.get('BOT_MD', 0) for c in comps]):,} ft"
                summary += f" ({depth_range})"
            
            summary += f"\n\n🎯 ZONE CHARACTERISTICS"
            if any('csg' in comp.get('ICON_NAME', '').lower() for comp in depth_data):
                summary += f"\n• Casing/conductor zone"
            if any('perfo' in comp.get('ICON_NAME', '').lower() for comp in depth_data):
                summary += f"\n• Productive zone with perforations"
            if any('esp' in comp.get('ICON_NAME', '').lower() for comp in depth_data):
                summary += f"\n• Artificial lift zone (ESP)"
            
            summary += f"\n\n💡 OPERATIONAL RECOMMENDATIONS"
            summary += f"\n• Focus monitoring on critical zones"
            summary += f"\n• Interval inspection per component type"
            summary += f"\n• Depth-specific maintenance planning"
        
        return summary