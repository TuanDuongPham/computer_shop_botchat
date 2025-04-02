import chromadb
from chromadb.utils import embedding_functions
from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
import re
import uuid
from services.enhance_product_embedding import generate_enhanced_product_document


class ChromaDB:
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_policy = None
        self.chunk_size = 512
        self.chunk_overlap = 128

    def connect(self, collection_name="computer_parts"):
        self.client = chromadb.PersistentClient()
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 100
            }
        )

        print(f"Connected to ChromaDB with collection: {collection_name}")
        return self

    def _create_chunks(self, text, metadata, product_id):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        chunk_ids = []
        chunk_metadatas = []

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Add sentence to current chunk if it fits
            if current_length + len(sentence) <= self.chunk_size or not current_chunk:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                # Save the current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Create a unique ID for this chunk
                chunk_id = f"{product_id}-{str(uuid.uuid4())[:8]}"
                chunk_ids.append(chunk_id)

                # Add product_id to metadata to track the source
                chunk_metadata = metadata.copy()
                chunk_metadata["product_id"] = str(product_id)
                chunk_metadata["chunk_index"] = len(chunks) - 1
                chunk_metadatas.append(chunk_metadata)

                # Start a new chunk with overlap
                overlap_tokens = current_chunk[-self.chunk_overlap:] if self.chunk_overlap < len(
                    current_chunk) else current_chunk
                current_chunk = overlap_tokens + [sentence]
                current_length = sum(len(s) for s in current_chunk)

        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            chunk_id = f"{product_id}-{str(uuid.uuid4())[:8]}"
            chunk_ids.append(chunk_id)

            chunk_metadata = metadata.copy()
            chunk_metadata["product_id"] = str(product_id)
            chunk_metadata["chunk_index"] = len(chunks) - 1
            chunk_metadatas.append(chunk_metadata)

        return chunks, chunk_ids, chunk_metadatas

    def add_product(self, product_id, product, category, specs_text):
        product_text = generate_enhanced_product_document(
            product, category, specs_text)

        metadata = {
            "category": category,
            "price": product["price"],
            "brand": product["brand"],
            "product_id": str(product_id),
            "model": product["model"]
        }

        # Create overlapping chunks
        chunks, chunk_ids, chunk_metadatas = self._create_chunks(
            product_text, metadata, product_id)

        # Add chunks to collection
        if chunks:
            self.collection.add(
                ids=chunk_ids,
                metadatas=chunk_metadatas,
                documents=chunks
            )

            self.collection.add(
                ids=[str(product_id)],
                metadatas=[metadata],
                documents=[product_text]
            )

    def search(self, query, n_results=3, filter_dict=None):
        chunk_results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 3,
            where=filter_dict
        )

        # Get unique product_ids from retrieved chunks
        product_ids = set()
        for metadata in chunk_results['metadatas'][0]:
            if 'product_id' in metadata:
                product_ids.add(metadata['product_id'])

        # Retrieve full products for these IDs
        if product_ids:
            where_filter = {"product_id": {"$in": list(product_ids)}}
            if filter_dict:
                where_filter = {"$and": [where_filter, filter_dict]}

            product_results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            return product_results

        return chunk_results

    def close(self):
        if self.client:
            if hasattr(self.client, 'persist'):
                self.client.persist()
            self.client = None
            self.collection = None
        print("ChromaDB connection closed")
