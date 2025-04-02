from database.chroma import ChromaDB
from database.postgres import PostgresDB
from generators.product_generator import ProductGenerator
from vietnamese_llm_helper import VietnameseLLMHelper


def main():
    postgres_db = PostgresDB().connect()
    chroma_db = ChromaDB().connect()

    vi_helper = VietnameseLLMHelper()

    try:
        # Generate Products data and add to Chroma DB
        generator = ProductGenerator(postgres_db, chroma_db)
        generator.generate_products()

        # Test Chroma DB Query
        query = "cpu intel giá trên 10 triệu"
        enhanced_query = vi_helper.enhance_vietnamese_query(query)
        results = chroma_db.search(
            enhanced_query, n_results=3, filter_dict=None)
        print(enhanced_query, results)

    finally:
        postgres_db.close()
        chroma_db.close()


if __name__ == "__main__":
    main()
