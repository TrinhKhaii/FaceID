from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
from typing import Optional, Tuple, Dict, List
import uuid

class QdrantFaceBank:
    def __init__(self, collection_name="FaceID", host="localhost", port=6333, embedding_size=512):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.embedding_size = embedding_size
        
    
    def add_user(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        point_id = str(uuid.uuid4())

        payload = {"name": name}
        if metadata:
            payload.update(metadata)

        vector = embedding.flatten().tolist()

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
    
    def recognize_and_adaptive_update(
        self, 
        test_emb: np.ndarray, 
        high_threshold: float = 0.7, 
        low_threshold: float = 0.5,
        name_filter: Optional[str] = None
    ) -> Tuple[str, float, bool, Dict]:
        query_vector = test_emb.flatten().tolist()
        
        query_filter = None
        if name_filter:
            query_filter = Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name_filter))]
            )

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=50
        ).points
        
        if not search_results:
            return "Unknown", 0.0, False, {}

        best_result = search_results[0]
        best_name = best_result.payload["name"]
        max_score = best_result.score
        
        all_scores = {}
        for result in search_results:
            name = result.payload["name"]
            score = result.score
            if name not in all_scores:
                all_scores[name] = []
            all_scores[name].append(score)
        
        updated = False
        
        if max_score < low_threshold:
            print(f"Unknown (score: {max_score:.3f} < {low_threshold})")
            return "Unknown", max_score, False, all_scores
        
        elif max_score >= high_threshold:
            print(f"{best_name} (score: {max_score:.3f}) - No need update")
            return best_name, max_score, False, all_scores
        
        else:
            self.add_user(best_name, test_emb, metadata={"updated": True})
            print(f"{best_name} (score: {max_score:.3f}) - UPDATED")
            return best_name, max_score, True, all_scores
    
    def get_user_embedding_count(self, name: str) -> int:
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name))]
            ),
            limit=10000,
            with_vectors=False,
            with_payload=False
        )
        return len(result[0])
    
    def get_all_users(self) -> List[str]:
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        names = set(point.payload["name"] for point in results)
        return sorted(list(names))
    
    def delete_user(self, name: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name))]
            )
        )
        print(f"Deleted all embeddings for {name}")
    
    def clear_all(self):
        self.client.delete_collection(collection_name=self.collection_name)
        self._create_collection_if_not_exists()
        print(f"Cleared collection: {self.collection_name}")
