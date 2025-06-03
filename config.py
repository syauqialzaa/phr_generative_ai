import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DCA API Configuration
DCA_API_BASE_URL = os.getenv("DCA_API_URL", "https://5c959a7dff3c.ngrok.app")
DCA_APP_BASE_URL = "https://syauqialzaa.github.io/dca"

# Vector store configuration
MILVUS_DB_PATH = "./milvus_dca.db"

# Server configuration
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", 8001))

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