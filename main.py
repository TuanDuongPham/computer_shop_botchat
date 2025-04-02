from database.chroma import ChromaDB
from database.postgres import PostgresDB
from generators.product_generator import ProductGenerator
from services.vietnamese_llm_helper import VietnameseLLMHelper
from services.policy_embedding import PolicyEmbeddingService
from services.policy_search import PolicySearchService
from services.enhance_search import EnhancedSearchService
import subprocess
import os


def main():
    # postgres_db = PostgresDB().connect()
    # chroma_db_products = ChromaDB().connect(collection_name="computer_parts")
    # chroma_db_policies = ChromaDB().connect(collection_name="policies")

    vi_helper = VietnameseLLMHelper()

    # Generate Products data and add to Chroma DB
    # generator = ProductGenerator(postgres_db, chroma_db)
    # generator.generate_products()

    # Embed policy
    # policy_file_path = "resources/policy.txt"
    # policy_service = PolicyEmbeddingService(chroma_db_policies)
    # policy_service.process_policy_file(policy_file_path)

    # Test Chroma DB Product Query
    # query = "Chip intel mạnh nhất tầm gía dưới 10 triệu"
    # enhanced_query = vi_helper.enhance_vietnamese_query(query)
    # search_product = EnhancedSearchService()
    # results = search_product.search(
    #     query=enhanced_query, n_results=3)
    # print(enhanced_query, results)

    # Test Chroma DB Policy Query
    query = "Phương thức thanh toán"
    policy_search = PolicySearchService()
    search_results = policy_search.search_policy(
        query=query,
        language='vi',
        n_results=3
    )

    policy_info = policy_search.format_policy_response(search_results)
    print(policy_info)

    # Run chat app
    # subprocess.Popen("start cmd /k fastapi dev app/server.py", shell=True)
    # subprocess.Popen("start cmd /k streamlit run app/app.py", shell=True)e()


if __name__ == "__main__":
    main()
