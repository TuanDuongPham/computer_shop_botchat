from src.database.chroma import ChromaDB
from src.services.enhance_search import EnhancedSearchService
from src.services.vietnamese_llm_helper import VietnameseLLMHelper
from src.services.shared_state import SharedStateService
from agents import Agent, Runner, FunctionTool, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from src.config import OPENAI_MODEL, OPENAI_API_KEY


class ProductAdvisorAgent:
    def __init__(self):
        self.vi_helper = VietnameseLLMHelper()
        self.search_service = EnhancedSearchService()
        self.shared_state = SharedStateService()
        self.model_client = OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY)
        )

        # Create agent using OpenAI Agent SDK
        self.agent = Agent(
            name="ProductAdvisor",
            model=self.model_client,
            handoff_description="Specialist agent for Products Advisor",
            handoffs=[self.handle_query],
            instructions="""Bạn là chuyên gia tư vấn linh kiện máy tính của cửa hàng TechPlus.
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

    async def search_products(self, query: str, language: str = "vi", n_results: int = 3):
        return self.search_service.search(query, language, n_results)

    async def handle_query(self, query: str, language: str = "vi"):
        try:
            # Step 1: Search for products based on query
            search_results = await self.search_products(query, language, n_results=5)

            if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
                return "Xin lỗi, tôi không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn. Bạn có thể mô tả chi tiết hơn không?"

            # Step 2: Format product results
            formatted_products = []
            advised_products = []

            for i, doc in enumerate(search_results['documents'][0]):
                if i >= len(search_results['metadatas'][0]):
                    break

                metadata = search_results['metadatas'][0][i]
                product_full_name = metadata.get('product_name', '')

                if not product_full_name:
                    product_brand = metadata.get('brand', '')
                    product_model = metadata.get('model', '')
                    product_full_name = f"{product_brand} {product_model}".strip(
                    )

                product_price = metadata.get('price', 0)
                product_category = metadata.get('category', '')

                advised_products.append({
                    "name": product_full_name,
                    "price": float(product_price),
                    "category": product_category,
                    "quantity": 1
                })

                specs_text = doc.split("SPECIFICATIONS:")[
                    1].strip() if "SPECIFICATIONS:" in doc else ""
                specs_lines = [line.strip()
                               for line in specs_text.split("\n") if line.strip()]
                specs_summary = ". ".join(specs_lines[:5])

                formatted_product = {
                    "name": product_full_name,
                    "category": product_category,
                    "price": product_price,
                    "specs_summary": specs_summary
                }

                formatted_products.append(formatted_product)

            # Lưu sản phẩm đã tư vấn vào shared state
            self.shared_state.set_recently_advised_products(advised_products)

            # Step 3: Create a context for the LLM with product information
            products_context = ""
            for i, product in enumerate(formatted_products, 1):
                from src.services.price_utils import format_price_usd_to_vnd
                products_context += f"""
                Sản phẩm {i}: {product['name']}
                Danh mục: {product['category']}
                Giá: {format_price_usd_to_vnd(product['price'])}
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

            # Step 5: Generate response using the agent
            response = await Runner.run(
                self.agent,
                [
                    {"role": "system", "content": self.agent.instructions},
                    {"role": "user", "content": prompt}
                ],
            )

            return response.final_output

        except Exception as e:
            print(f"Error in ProductAdvisorAgent.handle_query: {e}")
            return f"Xin lỗi, tôi đang gặp sự cố khi tìm kiếm thông tin sản phẩm. Vui lòng thử lại sau hoặc mô tả sản phẩm bạn cần theo cách khác."
