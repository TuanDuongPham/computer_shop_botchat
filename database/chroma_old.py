import chromadb
from chromadb.utils import embedding_functions
from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


class ChromaDB:
    def __init__(self):
        self.client = None
        self.collection = None

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

    def add_product(self, product_id, product, category, specs_text):
        product_text = f"{product['name']} - {product['brand']} {product['model']}. " \
            f"Category: {category}. " \
            f"Price: ${product['price']}. " \
            f"Specifications: {specs_text}. " \

        self.collection.add(
            ids=[str(product_id)],
            metadatas=[{
                "category": category,
                "price": product["price"],
                "brand": product["brand"]
            }],
            documents=[product_text]
        )

    def search(self, query, n_results=3, filter_dict=None):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )
        return results

    def close(self):
        if self.client:
            if hasattr(self.client, 'persist'):
                self.client.persist()
            self.client = None
            self.collection = None
        print("ChromaDB connection closed")
