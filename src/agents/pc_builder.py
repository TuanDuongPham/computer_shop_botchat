from src.database.chroma import ChromaDB
from src.services.enhance_search import EnhancedSearchService
from src.services.vietnamese_llm_helper import VietnameseLLMHelper
from src.services.shared_state import SharedStateService
from agents import Agent, Runner, FunctionTool, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from src.config import OPENAI_MODEL, OPENAI_API_KEY
import re


class PCBuilderAgent:
    def __init__(self):
        self.vi_helper = VietnameseLLMHelper()
        self.search_service = EnhancedSearchService()
        self.shared_state = SharedStateService()
        self.model_client = OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY)
        )

        self.pc_purposes = {
            "gaming": "Gaming",
            "office": "Văn phòng",
            "graphics": "Đồ họa/Sáng tạo nội dung",
            "dev": "Lập trình/Phát triển",
            "streaming": "Streaming",
            "general": "Đa năng"
        }

        self.agent = Agent(
            name="PCBuilder",
            model=self.model_client,
            handoff_description="Specialist agent for PC building advisor",
            handoffs=[self.handle_query, self.search_components],
            instructions="""Bạn là chuyên gia tư vấn xây dựng cấu hình máy tính của cửa hàng TechPlus.
            Nhiệm vụ của bạn là tư vấn, gợi ý và xây dựng cấu hình PC phù hợp dựa trên nhu cầu và ngân sách của khách hàng.
            
            Khi tư vấn xây dựng PC, bạn cần:
            1. Xác định rõ nhu cầu sử dụng của khách hàng (gaming, đồ họa, văn phòng, đa năng...)
            2. Xác định ngân sách mà khách hàng có thể chi cho cấu hình
            3. Đề xuất các linh kiện phù hợp và cân đối với nhau
            4. Giải thích lý do chọn từng linh kiện và ưu điểm của chúng
            5. Đảm bảo tính tương thích giữa các linh kiện
            6. Cung cấp tổng chi phí ước tính
            
            Quy tắc quan trọng:
            1. Tập trung vào nhu cầu thực sự của khách hàng, đừng đề xuất quá mức cần thiết
            2. Đảm bảo cân đối giữa các linh kiện, không có "bottleneck" (thắt cổ chai)
            3. Luôn kiểm tra tính tương thích giữa CPU và Motherboard (socket phải khớp nhau)
            4. Ưu tiên: CPU, GPU, RAM, Motherboard - bốn thành phần này quyết định hầu hết hiệu năng
            5. Đảm bảo nguồn điện (PSU) phù hợp với tổng công suất của hệ thống
            
            Các thành phần chính của một PC đầy đủ:
            - CPU (bộ xử lý)
            - Motherboard (bo mạch chủ)
            - RAM (bộ nhớ)
            - GPU (card đồ họa)
            - Storage (ổ cứng SSD/HDD)
            - PSU (nguồn)
            - Case (vỏ máy tính)
            - Cooling (tản nhiệt)
            
            Bạn có thể tìm kiếm trong cơ sở dữ liệu sản phẩm của cửa hàng để đề xuất các linh kiện cụ thể với giá thành chính xác.
            """,
        )

        # Agent đặc biệt để tìm kiếm thông tin từ database
        self.search_agent = Agent(
            name="ComponentSearcher",
            model=self.model_client,
            instructions="""Bạn là một chuyên gia tìm kiếm linh kiện máy tính.
            Nhiệm vụ của bạn là tìm kiếm các linh kiện phù hợp từ cơ sở dữ liệu sản phẩm.
            
            Bạn sẽ nhận yêu cầu tìm kiếm cho một loại linh kiện cụ thể, cùng với mô tả chi tiết về mục đích sử dụng và ngân sách.
            Hãy sử dụng các từ khóa phù hợp để tìm kiếm sản phẩm, kết hợp tên loại linh kiện với các thông số kỹ thuật.
            
            Luôn trả về một danh sách JSON có cấu trúc chuẩn, bao gồm các trường:
            - name: Tên đầy đủ của sản phẩm
            - price: Giá sản phẩm (số thực)
            - category: Danh mục sản phẩm
            - details: Mô tả chi tiết
            """
        )

    def _extract_budget(self, query):
        patterns = [
            r'(\d+)[\s]*tri[ệ|e]u[\s]*r?[ư|u][ỡ|õ]i',  # triệu rưỡi
            r'(\d+)[\s]*tri[ệ|e]u',  # triệu
            r'(\d+)[\s]*tr',  # tr
            r'(\d+)[\s]*M',  # 20M (million)
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                budget_value = int(match.group(1))
                budget_vnd = budget_value * 1000000

                if "rưỡi" in query or "ruoi" in query.lower() or "rưởi" in query:
                    budget_vnd += 500000

                return budget_vnd

        return None

    def _extract_purpose(self, query):
        query_lower = query.lower()

        purpose_keywords = {
            "gaming": ["gaming", "game", "chơi game", "fps", "battle", "esport", "pubg", "lol", "chơi gta"],
            "office": ["văn phòng", "excel", "word", "office", "làm việc", "word", "powerpoint"],
            "graphics": ["đồ họa", "photoshop", "illustrator", "premiere", "after effects", "thiết kế", "render", "3d", "blender", "vray"],
            "dev": ["lập trình", "code", "coding", "development", "dev", "phát triển", "web", "app", "database"],
            "streaming": ["stream", "streaming", "youtube", "phát sóng", "obs", "quay video", "content"],
            "general": ["đa năng", "đa dụng", "nhiều việc", "học tập", "giải trí"]
        }

        found_purposes = []
        for purpose, keywords in purpose_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                found_purposes.append(purpose)

        if len(found_purposes) > 0:
            return found_purposes
        else:
            return ["general"]

    async def search_components(self, category, search_query, budget_hint=None, n_results=3):
        """Tìm kiếm linh kiện từ database dựa trên danh mục và truy vấn tìm kiếm."""
        try:
            filter_dict = {"category": category}

            # Thêm gợi ý về ngân sách vào truy vấn nếu có
            if budget_hint:
                from src.services.price_utils import convert_usd_to_vnd
                # Chuyển đổi ngân sách từ VND sang USD (vì giá trong database lưu bằng USD)
                budget_usd = budget_hint / 25000  # Tỷ giá ước tính
                enhanced_query = f"{search_query} price range {budget_usd}"
            else:
                enhanced_query = search_query

            # Tìm kiếm trong database
            search_results = self.search_service.search(
                enhanced_query,
                language="vi",
                n_results=n_results,
                filters=filter_dict
            )

            # Định dạng kết quả trả về
            formatted_results = []
            if search_results and 'documents' in search_results and search_results['documents'][0]:
                for i, doc in enumerate(search_results['documents'][0]):
                    if i >= len(search_results['metadatas'][0]):
                        break

                    metadata = search_results['metadatas'][0][i]
                    product_full_name = metadata.get('product_name', '')

                    if not product_full_name:
                        brand = metadata.get('brand', '')
                        model = metadata.get('model', '')
                        product_full_name = f"{brand} {model}".strip()

                    price = float(metadata.get('price', 0))

                    # Trích xuất thông số kỹ thuật
                    specs_text = ""
                    if "SPECIFICATIONS:" in doc:
                        specs_text = doc.split("SPECIFICATIONS:")[1].strip()

                    formatted_results.append({
                        "name": product_full_name,
                        "price": price,
                        "category": category,
                        "details": specs_text
                    })

            return formatted_results
        except Exception as e:
            print(f"Error searching components: {e}")
            return []

    async def handle_query(self, query: str, language: str = "vi"):
        try:
            # Trích xuất thông tin từ yêu cầu của khách hàng
            budget = self._extract_budget(query)
            purposes = self._extract_purpose(query)

            # Xử lý ngân sách mặc định nếu không có
            if not budget:
                budget = 20000000  # 20 triệu VND mặc định
                budget_text = "không đề cập cụ thể"
            else:
                budget_text = f"{budget:,}đ".replace(",", ".")

            # Xử lý mục đích sử dụng
            purpose_text = ", ".join([self.pc_purposes[p]
                                      for p in purposes if p in self.pc_purposes])

            # Chuẩn bị enhanced query từ yêu cầu của khách hàng
            enhanced_query = self.vi_helper.enhance_vietnamese_query(query)

            # Chuẩn bị prompt để LLM tìm kiếm và tư vấn cấu hình PC
            prompt = f"""
            Bạn là chuyên gia tư vấn cấu hình PC tại cửa hàng TechPlus. Một khách hàng đã yêu cầu: "{query}"
            
            Sau khi phân tích yêu cầu, tôi xác định:
            - Ngân sách: {budget_text}
            - Mục đích sử dụng chính: {purpose_text}
            
            Dựa trên thông tin trên, hãy giúp tôi xây dựng một cấu hình PC phù hợp. Với mỗi linh kiện, hãy:
            1. Xác định tiêu chí quan trọng cho loại linh kiện đó (dựa trên mục đích sử dụng)
            2. Tìm kiếm trong cơ sở dữ liệu để tìm sản phẩm phù hợp
            3. Đưa ra đề xuất và giải thích tại sao sản phẩm đó phù hợp
            
            Sau đây là các loại linh kiện chính bạn cần tư vấn:
            - CPU (Bộ xử lý)
            - Motherboard (Bo mạch chủ)
            - RAM (Bộ nhớ)
            - GPU (Card đồ họa)
            - Storage (Ổ cứng)
            - PSU (Nguồn)
            - Case (Vỏ máy tính)
            - Cooling (Tản nhiệt)
            
            Đối với mỗi loại linh kiện, hãy tìm kiếm sản phẩm phù hợp trong cơ sở dữ liệu của chúng ta.
            Hãy phân bổ ngân sách hợp lý dựa trên mục đích sử dụng, ví dụ:
            - Với PC Gaming: Tập trung vào GPU, CPU mạnh, RAM đủ lớn.
            - Với PC Đồ họa: Cân bằng giữa CPU và GPU, RAM lớn, ổ cứng nhanh.
            - Với PC Văn phòng: CPU đủ dùng, RAM hợp lý, không cần GPU mạnh.
            
            Lưu ý đặc biệt:
            - Đảm bảo tính tương thích giữa các linh kiện (đặc biệt là CPU và Motherboard)
            - Ưu tiên các sản phẩm trong tầm giá và hiệu năng hợp lý
            - Tổng chi phí không nên vượt quá ngân sách của khách hàng, hoặc chỉ vượt một chút nếu thực sự cần thiết
            
            Quan trọng: Hãy FORMAT câu trả lời theo cấu trúc sau để dễ dàng trích xuất thông tin:
            - Đối với mỗi thành phần, hãy sử dụng tiêu đề "### [Tên thành phần]" (ví dụ: "### CPU")
            - Sau tiêu đề, đặt tên đầy đủ của sản phẩm được đề xuất trên một dòng riêng (ví dụ: "Intel Core i5-13600K")
            - Đặt giá của sản phẩm ở dạng "- Giá: XXX.XXXđ" trên một dòng riêng
            - Sau đó là phần giải thích và lý do chọn sản phẩm
            
            Hãy đảm bảo rằng mỗi thành phần đều có đầy đủ các thông tin trên theo đúng định dạng này.
            """

            # Chuẩn bị danh sách để lưu thông tin cấu hình
            pc_components = []

            # Tìm kiếm thông tin từng loại linh kiện
            component_categories = [
                "CPU", "Motherboard", "RAM", "GPU", "Storage", "PSU", "Case", "Cooling"]

            # Tạo cấu trúc đầu vào cho LLM với các kết quả tìm kiếm
            component_searches = {}

            # Thực hiện tìm kiếm song song cho tất cả các danh mục
            for category in component_categories:
                # Tạo truy vấn tìm kiếm dựa trên danh mục và mục đích sử dụng
                purpose_keywords = " ".join(
                    [self.pc_purposes[p] for p in purposes if p in self.pc_purposes])
                search_query = f"{category} for {purpose_keywords}"

                # Tính toán ngân sách tương đối cho từng loại linh kiện
                category_budget = None
                if budget:
                    if category == "CPU":
                        category_budget = budget * 0.2  # ~20% ngân sách
                    elif category == "GPU" and "gaming" in purposes:
                        category_budget = budget * 0.3  # ~30% ngân sách cho gaming
                    elif category == "GPU":
                        category_budget = budget * 0.25  # ~25% cho mục đích khác
                    elif category == "RAM":
                        category_budget = budget * 0.15  # ~15% ngân sách
                    elif category == "Motherboard":
                        category_budget = budget * 0.15  # ~15% ngân sách
                    elif category == "Storage":
                        category_budget = budget * 0.1  # ~10% ngân sách
                    elif category == "PSU":
                        category_budget = budget * 0.08  # ~8% ngân sách
                    elif category == "Case":
                        category_budget = budget * 0.05  # ~5% ngân sách
                    elif category == "Cooling":
                        category_budget = budget * 0.02  # ~2% ngân sách

                # Tìm kiếm các linh kiện phù hợp
                components = await self.search_components(category, search_query, category_budget, n_results=3)
                component_searches[category] = components

            # Bổ sung thông tin tìm kiếm vào prompt
            prompt += "\n\nKết quả tìm kiếm trong cơ sở dữ liệu của chúng ta:\n"

            for category, components in component_searches.items():
                prompt += f"\n{category} - Kết quả tìm kiếm:\n"
                if components:
                    for i, comp in enumerate(components, 1):
                        # Format giá tiền
                        from src.services.price_utils import format_price_usd_to_vnd
                        price_vnd = format_price_usd_to_vnd(comp['price'])

                        # Trích xuất các thông số kỹ thuật quan trọng
                        details = comp['details']
                        if len(details) > 200:
                            details = details[:200] + "..."

                        prompt += f"{i}. {comp['name']} - {price_vnd}\n   Thông số: {details}\n"
                else:
                    prompt += "Không tìm thấy sản phẩm phù hợp.\n"

            # Thêm hướng dẫn cuối cùng
            prompt += """
            Dựa trên các kết quả tìm kiếm trên, hãy xây dựng một cấu hình PC hoàn chỉnh và tối ưu. 
            Nếu một số linh kiện không có kết quả tìm kiếm, hãy đề xuất thông tin chung.
            
            Hãy sử dụng các kết quả tìm kiếm này để đề xuất cấu hình tốt nhất phù hợp với nhu cầu của khách hàng.
            Đưa ra đề xuất cuối cùng với danh sách đầy đủ các linh kiện, giá tiền, và tổng chi phí.
            
            Nhớ rằng, mỗi thành phần phải có tiêu đề dạng "### [Tên thành phần]", sau đó là tên sản phẩm trên một dòng riêng, 
            và giá sản phẩm dạng "- Giá: XXX.XXXđ" trên dòng tiếp theo.
            """

            # Gọi LLM để xử lý yêu cầu
            response = await Runner.run(
                self.agent,
                [
                    {"role": "system", "content": self.agent.instructions},
                    {"role": "user", "content": prompt}
                ]
            )

            final_response = response.final_output

            # Cải thiện phương pháp trích xuất sản phẩm từ phản hồi
            advised_products = []

            # Sử dụng regex để tìm các phần thông tin sản phẩm
            import re

            # Pattern để tìm các section
            section_pattern = r'### (CPU|Motherboard|RAM|GPU|Storage|PSU|Case|Cooling)[\s\S]*?(?=### |\Z)'
            sections = re.findall(section_pattern, final_response)

            # Trích xuất thông tin từng sản phẩm
            for category in component_categories:
                # Tìm section cho category hiện tại
                section_match = re.search(
                    f'### {category}([\s\S]*?)(?=### |\Z)', final_response)
                if section_match:
                    section_text = section_match.group(1).strip()

                    # Lấy dòng đầu tiên làm tên sản phẩm
                    lines = section_text.split('\n')
                    if lines:
                        product_name = lines[0].strip()

                        # Tìm giá sản phẩm
                        price_match = re.search(
                            r'[Gg]iá:?\s*(\d{1,3}(?:\.\d{3})*(?:\,\d+)?)\s*(?:đ|₫|VND)', section_text)
                        product_price = 0

                        if price_match:
                            price_str = price_match.group(1).replace(".", "")
                            try:
                                # Chuyển đổi giá từ VND sang USD
                                from src.services.price_utils import parse_usd_from_vnd
                                product_price = float(price_str)
                                product_price = parse_usd_from_vnd(
                                    product_price)
                            except:
                                product_price = 0

                        # Thêm sản phẩm vào danh sách (nếu có tên)
                        if product_name and not product_name.startswith('-') and not product_name.startswith('*'):
                            advised_products.append({
                                "name": product_name,
                                "price": product_price,
                                "category": category,
                                "quantity": 1
                            })

            # In thông tin debug
            print(f"Số lượng sản phẩm đã trích xuất: {len(advised_products)}")
            for product in advised_products:
                print(
                    f"- {product['category']}: {product['name']} - ${product['price']}")

            # Lưu sản phẩm đã tư vấn vào shared state
            if advised_products:
                self.shared_state.set_recently_advised_products(
                    advised_products)
                print(
                    f"Đã lưu {len(advised_products)} sản phẩm vào shared state")

            return final_response

        except Exception as e:
            print(f"Error in PCBuilderAgent.handle_query: {e}")
            import traceback
            traceback.print_exc()
            return f"Xin lỗi, tôi đang gặp sự cố khi xây dựng cấu hình PC. Vui lòng thử lại với yêu cầu rõ ràng hơn về ngân sách và mục đích sử dụng, ví dụ: 'Xây dựng PC gaming 25 triệu' hoặc 'PC đồ họa 30tr'."
