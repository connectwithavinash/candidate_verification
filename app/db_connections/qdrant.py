import os
from pyexpat import model
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import asyncio
from dotenv import load_dotenv

load_dotenv()

class QdrantConnection:
    
    def __init__(self) -> None:
        self.url = os.getenv('QDRANT_URL')
        self.port = os.getenv('QDRANT_PORT')
        self.client = AsyncQdrantClient(url=self.url, port=self.port, timeout=60)

    async def create_collection(self, collection_name):
        try:
            result = await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=512, 
                    distance=Distance.COSINE  # Use COSINE for FaceNet512
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000  # Use memory-mapped storage for large datasets
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=32,  # Number of connections per node (balances speed & accuracy)
                    ef_construct=400,  # Improves recall during indexing
                    full_scan_threshold=10000,  # Avoid brute-force scans
                    on_disk=True  # Store index on disk for large-scale datasets
                ),
                quantization_config=models.ScalarQuantization(  # Optional: Enable quantization for memory efficiency
                    scalar=models.ScalarQuantizationConfig(
                        type="int8", quantile=0.99, always_ram=True
                    )
                )
                # payload_schema={  # Define metadata schema for filtering
                #     "person_id": models.PayloadSchemaType.KEYWORD,
                #     "name": models.PayloadSchemaType.TEXT,
                #     "age": models.PayloadSchemaType.INTEGER,
                #     "timestamp": models.PayloadSchemaType.INTEGER
                # }
            )
            return result
            
        except Exception as e:
            print(f"Error creating collection: {e}")

    async def check_collection_existance(self, collection_name):
        try:
            result = await self.client.collection_exists(collection_name=f"{collection_name}")
            return result
        except Exception as e:
            print(e)

    async def delete_collection(self, collection_name):

        try:
            result = await self.client.delete_collection(collection_name=collection_name)
            return result

        except Exception as e:
            print(e)
    
    async def get_collection_info(self, collection_name):
        try:
            result = await self.client.get_collection(collection_name)
            return result    
        
        except Exception as e:
            print(f"Error getting collection info: {e}")

    async def list_all_collection(self):
        try:
            result = await self.client.get_collections()
            return result 
        
        except Exception as e:
            print(e)
    
    async def upload_points_to_collection(self,collection_name, payload):
        pass

    async def batch_upload_points_to_collection(self, collection_name, payload):
        try:
            operation_info = await self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=models.Batch(
                    ids=payload.get('ids'),
                    vectors=payload.get('vectors'),
                    payloads=payload.get('payload')
                )
            )
            return operation_info
        except Exception as e:
            print(f"Error uploading points: {e}")
    
    async def query_collection(self, collection_name, query_vector: list):
        try:
            search_result = await self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    with_payload=True,
                    limit=3
                )
            
            return search_result.points
        
        except Exception as e:
              print(f"Error querying collection: {e}")

    async def batch_upload_points_to_collection(self, collection_name, payload):
        try:
            operation_info = await self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=models.Batch(
                    ids=payload.get('ids'),
                    vectors=payload.get('vectors'),
                    payloads=payload.get('payload')
                )
            )
            return operation_info
        except Exception as e:
            print(f"Error uploading points: {e}")
    
    async def query_collection(self, collection_name, query_vector: list):
        try:
            search_result = await self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    with_payload=True,
                    limit=3
                )
            
            return search_result.points
        
        except Exception as e:
              print(f"Error querying collection: {e}")

qdrant_client = QdrantConnection()

if __name__ == "__main__":
    result = asyncio.run(qdrant_client.list_all_collection())
    print(result)