from src.database.chroma import ChromaDB
from src.services.enhance_search import EnhancedSearchService
from src.services.vietnamese_llm_helper import VietnameseLLMHelper
from agents import Agent, Runner, FunctionTool, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from src.config import OPENAI_MODEL, OPENAI_API_KEY
import re


class PCBuilderAgent:
    def __init__(self):
        self.vi_helper = VietnameseLLMHelper()
        self.search_service = EnhancedSearchService()
        self.model_client = OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY)
        )

        self.budget_ranges = {
            "entry": {"min": 8000000, "max": 15000000, "name": "Phổ thông"},
            "mid": {"min": 15000000, "max": 25000000, "name": "Tầm trung"},
            "high": {"min": 25000000, "max": 40000000, "name": "Cao cấp"},
            "extreme": {"min": 40000000, "max": 100000000, "name": "Enthusiast"}
        }

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
            handoffs=[self.handle_query],
            instructions="""Bạn là chuyên gia tư vấn xây dựng cấu hình máy tính của cửa hàng TechPlus.
            Nhiệm vụ của bạn là tư vấn, gợi ý và xây dựng cấu hình PC phù hợp dựa trên nhu cầu và ngân sách của khách hàng.
            
            Khi tư vấn xây dựng PC, bạn cần:
            1. Xác định rõ nhu cầu sử dụng của khách hàng (gaming, đồ họa, văn phòng, đa năng...)
            2. Xác định ngân sách mà khách hàng có thể chi cho cấu hình
            3. Đề xuất các linh kiện phù hợp và cân đối với nhau
            4. Giải thích lý do chọn từng linh kiện và ưu điểm của chúng
            5. Đảm bảo tính tương thích giữa các linh kiện
            6. Cung cấp tổng chi phí ước tính
            
            Hãy đảm bảo cấu hình được đề xuất:
            - Cân đối về hiệu năng (không bottleneck)
            - Phù hợp với mục đích sử dụng
            - Nằm trong ngân sách của khách hàng
            - Có tính đến khả năng nâng cấp trong tương lai
            
            Bạn có thể tìm kiếm trong cơ sở dữ liệu sản phẩm của cửa hàng để đề xuất các linh kiện cụ thể với giá thành chính xác.
            """,
        )

    async def search_products_by_category(self, category, query=None, n_results=3):
        filter_dict = {"category": category}

        search_query = query or category
        search_results = self.search_service.search(
            search_query,
            language="vi",
            n_results=n_results,
            filters=filter_dict
        )

        return search_results

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

    async def build_pc_config(self, budget, purposes):
        configs = {}
        total_cost = 0

        budget_allocation = {}

        budget_allocation = {
            "CPU": 0.20,
            "Motherboard": 0.15,
            "GPU": 0.25,
            "RAM": 0.10,
            "Storage": 0.10,
            "PSU": 0.08,
            "Case": 0.07,
            "Cooling": 0.05
        }

        if "gaming" in purposes:
            budget_allocation["GPU"] = 0.35
            budget_allocation["CPU"] = 0.20
            budget_allocation["RAM"] = 0.10
            budget_allocation["Motherboard"] = 0.12
            budget_allocation["Storage"] = 0.08
            budget_allocation["Case"] = 0.05
            budget_allocation["PSU"] = 0.07
            budget_allocation["Cooling"] = 0.03

        elif "graphics" in purposes:
            budget_allocation["CPU"] = 0.25
            budget_allocation["GPU"] = 0.30
            budget_allocation["RAM"] = 0.15
            budget_allocation["Storage"] = 0.12
            budget_allocation["Motherboard"] = 0.08
            budget_allocation["Case"] = 0.04
            budget_allocation["PSU"] = 0.04
            budget_allocation["Cooling"] = 0.02

        elif "office" in purposes:
            budget_allocation["CPU"] = 0.25
            budget_allocation["RAM"] = 0.15
            budget_allocation["Storage"] = 0.15
            budget_allocation["GPU"] = 0.10
            budget_allocation["Motherboard"] = 0.15
            budget_allocation["Case"] = 0.10
            budget_allocation["PSU"] = 0.07
            budget_allocation["Cooling"] = 0.03

        elif "dev" in purposes:
            budget_allocation["CPU"] = 0.30
            budget_allocation["RAM"] = 0.20
            budget_allocation["Storage"] = 0.15
            budget_allocation["GPU"] = 0.10
            budget_allocation["Motherboard"] = 0.12
            budget_allocation["PSU"] = 0.06
            budget_allocation["Case"] = 0.04
            budget_allocation["Cooling"] = 0.03

        for category, percentage in budget_allocation.items():
            component_budget = budget * percentage

            purpose_query = " ".join([self.pc_purposes[p]
                                     for p in purposes if p in self.pc_purposes])
            query = f"{category} for {purpose_query}"

            search_results = await self.search_products_by_category(category, query, n_results=5)

            if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
                configs[category] = {
                    "name": f"Không tìm thấy {category} phù hợp",
                    "price": 0
                }
                continue

            best_component = None
            smallest_price_diff = float('inf')

            for i, doc in enumerate(search_results['documents'][0]):
                if i >= len(search_results['metadatas'][0]):
                    break

                metadata = search_results['metadatas'][0][i]
                price = float(metadata.get('price', 0))

                if price <= component_budget:
                    price_diff = component_budget - price
                    if price_diff < smallest_price_diff:
                        smallest_price_diff = price_diff
                        best_component = {
                            "name": f"{metadata.get('brand', '')} {metadata.get('model', '')}",
                            "price": price,
                            "category": category,
                            "details": doc
                        }

            if not best_component and search_results['metadatas'][0]:
                cheapest_component = None
                cheapest_price = float('inf')

                for i, metadata in enumerate(search_results['metadatas'][0]):
                    price = float(metadata.get('price', 0))
                    if price < cheapest_price:
                        cheapest_price = price
                        cheapest_component = {
                            "name": f"{metadata.get('brand', '')} {metadata.get('model', '')}",
                            "price": price,
                            "category": category,
                            "details": search_results['documents'][0][i]
                        }

                best_component = cheapest_component

            if best_component:
                configs[category] = best_component
                total_cost += best_component["price"]
            else:
                configs[category] = {
                    "name": f"Không tìm thấy {category} phù hợp",
                    "price": 0
                }

        return {
            "configs": configs,
            "total_cost": total_cost
        }

    async def handle_query(self, query: str, language: str = "vi"):
        try:
            # Bước 1: Trích xuất thông tin từ câu hỏi
            budget = self._extract_budget(query)
            purposes = self._extract_purpose(query)

            if not budget:
                budget = 20000000

            # Bước 2: Xây dựng cấu hình PC
            pc_config = await self.build_pc_config(budget, purposes)

            # Bước 3: Chuẩn bị thông tin cho output
            config_details = ""
            for category, component in pc_config["configs"].items():
                component_price = "{:,.0f}".format(
                    component["price"]).replace(",", ".")
                config_details += f"\n### {category}\n"
                config_details += f"- {component['name']}\n"
                config_details += f"- Giá: {component_price}đ\n"

                if "details" in component:
                    specs_text = component["details"].split("SPECIFICATIONS:")[1].strip(
                    ) if "SPECIFICATIONS:" in component["details"] else ""
                    specs_lines = [line.strip() for line in specs_text.split(
                        "\n") if line.strip()][:3]
                    if specs_lines:
                        config_details += "- Thông số chính: " + \
                            "; ".join(specs_lines) + "\n"

            total_cost = "{:,.0f}".format(
                pc_config["total_cost"]).replace(",", ".")

            purpose_text = ", ".join([self.pc_purposes[p]
                                     for p in purposes if p in self.pc_purposes])

            # Bước 4: Chuẩn bị prompt cho phản hồi
            prompt = f"""
            Người dùng yêu cầu xây dựng cấu hình PC: "{query}"
            
            Tôi đã tạo một cấu hình PC dựa trên:
            - Ngân sách: {"{:,.0f}".format(budget).replace(",", ".")}đ
            - Mục đích sử dụng: {purpose_text}
            
            Cấu hình PC được đề xuất:
            {config_details}
            
            Tổng chi phí ước tính: {total_cost}đ
            
            Hãy trả lời người dùng với thông tin chi tiết về cấu hình này, giải thích lý do chọn từng linh kiện,
            và đề cập đến hiệu năng dự kiến cho các mục đích sử dụng của họ. Cung cấp lời khuyên về các nâng cấp 
            tiềm năng nếu có thêm ngân sách hoặc những gì có thể cắt giảm nếu ngân sách hạn hẹp hơn.
            
            Đảm bảo câu trả lời thân thiện, chuyên nghiệp và dễ hiểu.
            """

            # Bước 5: Tạo response từ agent
            response = await Runner.run(
                self.agent,
                [
                    {"role": "system", "content": self.agent.instructions},
                    {"role": "user", "content": prompt}
                ],
            )

            return response.final_output

        except Exception as e:
            print(f"Error in PCBuilderAgent.handle_query: {e}")
            return f"Xin lỗi, tôi đang gặp sự cố khi xây dựng cấu hình PC. Vui lòng thử lại với yêu cầu rõ ràng hơn về ngân sách và mục đích sử dụng, ví dụ: 'Xây dựng PC gaming 25 triệu' hoặc 'PC đồ họa 30tr'."
