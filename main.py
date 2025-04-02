from database.chroma import ChromaDB
from database.postgres import PostgresDB
from generators.product_generator import ProductGenerator
from services.vietnamese_llm_helper import VietnameseLLMHelper
from services.policy_embedding import PolicyEmbeddingService
from services.policy_search import PolicySearchService
import subprocess
import os


def main():
    postgres_db = PostgresDB().connect()
    chroma_db_products = ChromaDB().connect(collection_name="computer_parts")
    chroma_db_policies = ChromaDB().connect(collection_name="policies")

    vi_helper = VietnameseLLMHelper()

    try:
        # Generate Products data and add to Chroma DB
        # generator = ProductGenerator(postgres_db, chroma_db)
        # generator.generate_products()

        # Embed policy
        # policy_file_path = "resources/policy.txt"
        # policy_service = PolicyEmbeddingService(chroma_db_policies)
        # policy_service.process_policy_file(policy_file_path)

        # Test Chroma DB Product Query
        # query = "Nguồn 650W trên 2 triệu"
        # enhanced_query = vi_helper.enhance_vietnamese_query(query)
        # results = chroma_db_products.search(
        #     enhanced_query, n_results=3, filter_dict=None)
        # print(enhanced_query, results)

        # Test Chroma DB Policy Query
        # query = "Chi tiết chính sách đổi trả"
        # policy_search = PolicySearchService()
        # search_results = policy_search.search_policy(
        #     query=query,
        #     language='vi',
        #     n_results=3
        # )

        # # Extract raw policy information
        # policy_info = policy_search.format_policy_response(search_results)
        # subprocess.run(["streamlit", "run", "app/app.py"])
        # result = subprocess.run(["fastapi", "dev", "app/server.py"],
        #                         shell=True, capture_output=True, text=True)

        subprocess.run("start /wait fastapi dev app/server.py", shell=True)
        subprocess.run("start /wait streamlit run app/app.py", shell=True)

    finally:
        postgres_db.close()
        chroma_db_products.close()
        chroma_db_policies.close()


if __name__ == "__main__":
    main()
