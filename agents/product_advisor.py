from database.chroma import ChromaDB
from services.enhance_search import EnhancedSearchService
from services.vietnamese_llm_helper import VietnameseLLMHelper
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from config import OPENAI_MODEL, OPENAI_API_KEY


class ProductAdvisorAgent:
    def __init__(self):
        self.vi_helper = VietnameseLLMHelper()
        self.search_service = EnhancedSearchService()
        self.model_client = OpenAIChatCompletionClient(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
        )

        # Create Autogen assistant
        self.agent = AssistantAgent(
            name="ProductAdvisor",
            model_client=self.model_client,
            tools=[self.search_products, self.handle_query],
            system_message="""Bạn là chuyên gia tư vấn linh kiện máy tính của cửa hàng TechPlus.
            Nhiệm vụ của bạn là tư vấn, cung cấp thông tin chi tiết, và so sánh các linh kiện máy tính.
            Khi một khách hàng đưa ra yêu cầu, hãy phân tích nhu cầu của họ và đưa ra các lựa chọn phù hợp.
            
            Khi trả lời, hãy:
            1. Đảm bảo thông tin kỹ thuật chính xác
            2. So sánh ưu và nhược điểm giữa các sản phẩm
            3. Giải thích thuật ngữ kỹ thuật một cách dễ hiểu
            4. Đề xuất lựa chọn phù hợp nhất với nhu cầu và ngân sách
            
            Bạn có quyền truy cập vào cơ sở dữ liệu sản phẩm và có thể tìm kiếm các sản phẩm phù hợp với yêu cầu của khách hàng.
            """,
        )

    def search_products(self, query: str, language: str = "vi", n_results: int = 3):
        return self.search_service.search(query, language, n_results)

    def handle_query(self, query: str, language: str = "vi"):
        try:
            # Step 1: Search for products based on query
            search_results = self.search_products(query, language, n_results=5)

            if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
                return "Xin lỗi, tôi không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn. Bạn có thể mô tả chi tiết hơn không?"

            # Step 2: Format product results
            formatted_products = []

            for i, doc in enumerate(search_results['documents'][0]):
                if i >= len(search_results['metadatas'][0]):
                    break

                metadata = search_results['metadatas'][0][i]
                product_name = metadata.get('model', '')
                product_brand = metadata.get('brand', '')
                product_price = metadata.get('price', 0)
                product_category = metadata.get('category', '')

                # Extract basic specs from document
                specs_text = doc.split("SPECIFICATIONS:")[
                    1].strip() if "SPECIFICATIONS:" in doc else ""
                specs_lines = [line.strip()
                               for line in specs_text.split("\n") if line.strip()]
                specs_summary = ". ".join(specs_lines[:5])  # Get first 5 specs

                formatted_product = {
                    "name": f"{product_brand} {product_name}",
                    "category": product_category,
                    "price": product_price,
                    "specs_summary": specs_summary
                }

                formatted_products.append(formatted_product)

            # Step 3: Create a context for the LLM with product information
            products_context = ""
            for i, product in enumerate(formatted_products, 1):
                products_context += f"""
                Sản phẩm {i}: {product['name']}
                Danh mục: {product['category']}
                Giá: {format(int(product['price']), ',')}đ
                Thông số chính: {product['specs_summary']}
                ---
                """

            # Step 4: Prepare prompt for the response generation
            prompt = f"""
            Người dùng đang hỏi: "{query}"
            
            Dựa trên truy vấn, tôi đã tìm thấy các sản phẩm sau:
            
            {products_context}
            
            Hãy trả lời người dùng với thông tin chi tiết về các sản phẩm này. Nên đề cập đến:
            1. Tóm tắt các sản phẩm tìm thấy
            2. So sánh sản phẩm dựa trên thông số kỹ thuật và giá cả
            3. Đề xuất sản phẩm phù hợp nhất dựa vào truy vấn người dùng
            4. Giải thích các thuật ngữ kỹ thuật dễ hiểu nếu cần
            
            Định dạng câu trả lời rõ ràng, có cấu trúc dễ đọc.
            """

            # Step 5: Generate response using the agent's LLM
            from openai import OpenAI
            from config import OPENAI_API_KEY, OPENAI_MODEL

            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.agent.system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in ProductAdvisorAgent.handle_query: {e}")
            return f"Xin lỗi, tôi đang gặp sự cố khi tìm kiếm thông tin sản phẩm. Vui lòng thử lại sau hoặc mô tả sản phẩm bạn cần theo cách khác."
