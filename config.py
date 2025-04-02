import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    "dbname": os.environ.get("POSTGRES_DBNAME", "computer-shop"),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "admin"),
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432")
}

# ChromaDB Configuration
CHROMA_CLIENT_SETTINGS = {
    "chroma_db_impl": "duckdb+parquet",
    "persist_directory": os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
}

# Product Generation Settings
PRODUCTS_PER_CATEGORY = 100
BATCH_SIZE = 5
MAX_BATCH_ATTEMPTS = 30

# Product Categories
PRODUCT_CATEGORIES = [
    "CPU", "Motherboard", "RAM", "PSU",
    "GPU", "Storage", "Case", "Cooling"
]
