import numpy as np
from typing import List, Dict
import logging
from config import DCA_KNOWLEDGE_BASE

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.collection_name = "dca_knowledge"
        self.collection = None
        self.available = MILVUS_AVAILABLE
    
    def initialize(self):
        """Initialize vector store"""
        if not self.available:
            logger.warning("Milvus not available, vector store disabled")
            return
            
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
            self.available = False
    
    def _create_collection(self):
        """Create collection schema"""
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
        """Insert base knowledge into vector store"""
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
        """Search for similar knowledge"""
        if not self.available or not self.collection:
            return []
        
        try:
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
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_fallback_knowledge(self, query: str) -> List[Dict]:
        """Get fallback knowledge when vector store is not available"""
        query_lower = query.lower()
        results = []
        
        for item in DCA_KNOWLEDGE_BASE:
            # Simple keyword matching
            if any(keyword in query_lower for keyword in item["keywords"]):
                results.append({
                    "content": item["content"],
                    "category": item["category"],
                    "score": 1.0
                })
        
        return results[:3]