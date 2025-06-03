import numpy as np
from typing import List, Dict
import logging
from config import DCA_KNOWLEDGE_BASE, WELLBORE_KNOWLEDGE_BASE

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.collection_name = "knowledge_base"
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
        schema = CollectionSchema(fields, "DCA and Wellbore knowledge base")
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
        
        # Combine DCA and Wellbore knowledge
        all_knowledge = DCA_KNOWLEDGE_BASE + WELLBORE_KNOWLEDGE_BASE
        
        for item in all_knowledge:
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
        
        # Combine DCA and Wellbore knowledge for fallback
        all_knowledge = DCA_KNOWLEDGE_BASE + WELLBORE_KNOWLEDGE_BASE
        
        for item in all_knowledge:
            # Simple keyword matching
            if any(keyword in query_lower for keyword in item["keywords"]):
                results.append({
                    "content": item["content"],
                    "category": item["category"],
                    "score": 1.0
                })
        
        return results[:3]
    
    def add_dca_knowledge(self, content: str, category: str, keywords: List[str]) -> bool:
        """Add new DCA knowledge to the vector store"""
        if not self.available:
            return False
        
        try:
            # Generate embedding (simplified)
            embedding = np.random.rand(384).tolist()
            
            # Get next ID
            next_id = len(DCA_KNOWLEDGE_BASE) + len(WELLBORE_KNOWLEDGE_BASE) + 1
            
            # Insert into vector store
            data = [[next_id, embedding, content, category]]
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Added new knowledge: {category}")
            return True
        except Exception as e:
            logger.error(f"Error adding DCA knowledge: {e}")
            return False
    
    def add_wellbore_knowledge(self, content: str, category: str, keywords: List[str]) -> bool:
        """Add new wellbore knowledge to the vector store"""
        if not self.available:
            return False
        
        try:
            # Generate embedding (simplified)
            embedding = np.random.rand(384).tolist()
            
            # Get next ID
            next_id = len(DCA_KNOWLEDGE_BASE) + len(WELLBORE_KNOWLEDGE_BASE) + 1
            
            # Insert into vector store
            data = [[next_id, embedding, content, category]]
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Added new wellbore knowledge: {category}")
            return True
        except Exception as e:
            logger.error(f"Error adding wellbore knowledge: {e}")
            return False
    
    def search_by_category(self, category: str, top_k: int = 5) -> List[Dict]:
        """Search knowledge by specific category"""
        if not self.available or not self.collection:
            return self._get_fallback_by_category(category)
        
        try:
            # Use filter to search by category
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[[0.0] * 384],  # Dummy embedding
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=f'category == "{category}"',
                output_fields=["content", "category"]
            )
            
            return [{"content": hit.entity.get("content"), 
                     "category": hit.entity.get("category"),
                     "score": hit.score} for hit in results[0]]
        except Exception as e:
            logger.error(f"Error searching by category: {e}")
            return self._get_fallback_by_category(category)
    
    def _get_fallback_by_category(self, category: str) -> List[Dict]:
        """Get fallback knowledge by category"""
        results = []
        all_knowledge = DCA_KNOWLEDGE_BASE + WELLBORE_KNOWLEDGE_BASE
        
        for item in all_knowledge:
            if item["category"] == category:
                results.append({
                    "content": item["content"],
                    "category": item["category"],
                    "score": 1.0
                })
        
        return results
    
    def get_all_categories(self) -> List[str]:
        """Get list of all available knowledge categories"""
        all_knowledge = DCA_KNOWLEDGE_BASE + WELLBORE_KNOWLEDGE_BASE
        categories = list(set([item["category"] for item in all_knowledge]))
        return sorted(categories)
    
    def get_knowledge_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        dca_count = len(DCA_KNOWLEDGE_BASE)
        wellbore_count = len(WELLBORE_KNOWLEDGE_BASE)
        total_count = dca_count + wellbore_count
        
        categories = self.get_all_categories()
        
        return {
            "total_knowledge_items": total_count,
            "dca_knowledge_items": dca_count,
            "wellbore_knowledge_items": wellbore_count,
            "total_categories": len(categories),
            "categories": categories,
            "vector_store_available": self.available,
            "collection_name": self.collection_name
        }