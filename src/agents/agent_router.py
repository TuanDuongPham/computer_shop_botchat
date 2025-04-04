from typing import Dict, Any
from agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from src.config import OPENAI_API_KEY, OPENAI_MODEL
import json
import re


def extract_json_from_response(response_text):
    if "```json" in response_text:
        start_idx = response_text.find("```json") + 7
        end_idx = response_text.rfind("```")
        if end_idx > start_idx:
            return response_text[start_idx:end_idx].strip()

    elif "```" in response_text:
        start_idx = response_text.find("```") + 3
        end_idx = response_text.rfind("```")
        if end_idx > start_idx:
            return response_text[start_idx:end_idx].strip()

    import re
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}))*\}'
    json_match = re.search(json_pattern, response_text)
    if json_match:
        return json_match.group(0).strip()

    return response_text.strip()


class AgentRouter:
    def __init__(self, config=None):
        self.config = config or {}
        self.agent_types = {
            "product_advisor": "Tư vấn linh kiện máy tính",
            "policy_advisor": "Tư vấn chính sách cửa hàng",
            "pc_builder": "Tư vấn xây dựng PC",
            "order_processor": "Xác nhận đặt hàng và gửi email",
            "general": "Chào hỏi và hỏi đáp chung"
        }

        # self.keywords = {
        #     "product_advisor": [
        #         "cpu", "chip", "vi xử lý", "bộ xử lý", "core i", "intel", "amd", "ryzen", "xeon", "processor",
        #         "gpu", "card đồ họa", "vga", "nvidia", "radeon", "rtx", "gtx",
        #         "ram", "bộ nhớ", "memory", "ddr", "dimm",
        #         "bo mạch chủ", "mainboard", "main", "bo mẹ", "mother board",
        #         "ổ cứng", "ssd", "hdd", "nvme", "m.2", "sata",
        #         "nguồn", "power supply", "psu",
        #         "vỏ máy tính", "case", "thùng máy",
        #         "tản nhiệt", "fan", "quạt", "cooling", "aio"
        #     ],
        #     "policy_advisor": [
        #         "chính sách", "bảo hành", "warranty", "đổi trả", "hoàn tiền", "refund", "thanh toán",
        #         "payment", "giao hàng", "shipping", "delivery", "trả góp", "installment", "bảo mật",
        #         "quy định", "policy", "đổi", "trả", "hoàn", "vận chuyển"
        #     ],
        #     "pc_builder": [
        #         "build pc", "xây dựng pc", "lắp máy tính", "cấu hình pc", "cấu hình máy tính",
        #         "pc gaming", "pc đồ họa", "pc văn phòng", "build", "xây dựng cấu hình", "tư vấn cấu hình",
        #         "lắp ráp pc", "xây cấu hình", "dựng cấu hình", "lắp pc", "build máy", "lắp máy", "xây pc",
        #         "20 triệu", "25 triệu", "30 triệu", "15 triệu", "cấu hình", "lắp đặt pc"
        #     ],
        #     "order_processor": [
        #         "đặt hàng", "mua hàng", "order", "thanh toán", "purchase", "đơn hàng", "invoice",
        #         "mua", "đặt mua", "checkout", "giỏ hàng", "shopping cart", "đặt", "đơn đặt hàng",
        #         "mua ngay", "đặt giao", "tôi muốn mua", "tôi muốn đặt", "mua sản phẩm", "đặt sản phẩm",
        #         "gửi đến địa chỉ", "vận chuyển tới", "giao đến", "mua liền", "mua luôn"
        #     ]
        # }

        self.model_client = OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY)
        )

        # Create intent classifier agent
        self.intent_classifier = Agent(
            name="IntentClassifier",
            model=self.model_client,
            instructions="""
            Bạn là một AI phân loại ý định của người dùng trong cửa hàng máy tính. 
            Nhiệm vụ của bạn là phân loại đoạn text đầu vào chính xác vào một trong các danh mục sau đây:
            
            DANH MỤC PHÂN LOẠI:
            1. product_advisor: Các câu hỏi về thông tin, tư vấn, so sánh hoặc mua các linh kiện máy tính.
            - Liên quan đến: CPU, chip xử lý, vi xử lý, GPU, card đồ họa, VGA, RAM, Mainboard, bo mạch chủ, main, SSD, HDD, ổ cứng, PSU, nguồn máy tính, Case, vỏ máy tính, tản nhiệt, fan, quạt
            - Ví dụ: "RAM nào tốt cho chơi game", "CPU Intel nào phù hợp với ngân sách 5 triệu", "So sánh card RTX 4060 và 3070", "Tư vấn chip Intel cho PC gaming"
            
            2. policy_advisor: Các câu hỏi về chính sách cửa hàng, quy định, quy trình
            - Liên quan đến: bảo hành, đổi trả, hoàn tiền, thanh toán, trả góp, vận chuyển, giao hàng, bảo mật thông tin
            - Ví dụ: "Chính sách bảo hành là gì", "Làm thế nào để đổi trả", "Có hỗ trợ trả góp không"
            
            3. pc_builder: Yêu cầu xây dựng cấu hình PC hoàn chỉnh
            - Liên quan đến: xây dựng PC, tư vấn cấu hình, build PC, PC gaming, PC đồ họa, PC văn phòng
            - Ví dụ: "Xây dựng PC gaming 20 triệu", "Cần một bộ máy tính văn phòng", "Gợi ý cấu hình máy tính đồ họa"
            
            4. order_processor: Yêu cầu đặt hàng, xác nhận đơn hàng, thanh toán
            - Liên quan đến: đặt hàng, mua hàng, thanh toán, đơn hàng, giỏ hàng, checkout
            - Ví dụ: "Tôi muốn đặt một bộ PC", "Làm thế nào để thanh toán đơn hàng", "Tôi muốn mua một sản phẩm"
            
            5. general: Chào hỏi, hỏi đáp chung không liên quan đến 4 loại trên
            - Ví dụ: "Xin chào", "Cửa hàng mở cửa lúc mấy giờ", "Thông tin liên hệ của cửa hàng"
            
            QUY TRÌNH PHÂN LOẠI:
            1. Đọc kỹ nội dung câu hỏi của người dùng
            2. Xác định từ khóa chính trong câu hỏi (linh kiện, chính sách, cấu hình PC, đặt hàng)
            3. Phân loại vào danh mục phù hợp nhất (product_advisor, policy_advisor, pc_builder, order_processor, general)
            4. Gán mức độ tin cậy (confidence) từ 0.0 đến 1.0 cho phân loại
            
            QUAN TRỌNG:
            - Nếu câu hỏi về CPU, chip, intel, AMD, GPU, RAM, Mainboard, SSD/HDD, PSU, Case -> PHẢI xếp vào product_advisor
            - Nếu người dùng hỏi về đặc điểm, so sánh hoặc đề xuất linh kiện cụ thể -> PHẢI xếp vào product_advisor
            - Nếu câu hỏi chứa từ khóa "cấu hình PC", "build PC", "lắp máy tính" -> xếp vào pc_builder
            - Nếu câu hỏi chứa từ khóa "đặt hàng", "mua", "thanh toán", "giỏ hàng" -> xếp vào order_processor
            - Nếu câu hỏi về bảo hành, đổi trả, thanh toán, chính sách -> xếp vào policy_advisor
            
            Trả về kết quả dưới dạng JSON với format:
            {
                "intent": "<loại agent>",
                "confidence": <điểm tin cậy từ 0.0 đến 1.0>,
                "reasoning": "<giải thích ngắn gọn lý do>"
            }
            
            Đảm bảo giá trị "intent" là một trong các giá trị: "product_advisor", "policy_advisor", "pc_builder", "order_processor", "general"
            """,
        )

    # def _keyword_based_classification(self, user_query: str) -> Dict[str, Any]:
    #     user_query_lower = user_query.lower()

    #     special_keywords = {
    #         "chip": "product_advisor",
    #         "chip intel": "product_advisor",
    #         "chip amd": "product_advisor",
    #         "intel": "product_advisor",
    #         "core i": "product_advisor",
    #         "ryzen": "product_advisor",
    #         "xây dựng pc": "pc_builder",
    #         "build pc": "pc_builder",
    #         "build cấu hình": "pc_builder",
    #         "đặt hàng": "order_processor",
    #         "mua sản phẩm": "order_processor",
    #         "chính sách bảo hành": "policy_advisor"
    #     }

    #     for keyword, intent in special_keywords.items():
    #         if keyword in user_query_lower:
    #             return {
    #                 "intent": intent,
    #                 "confidence": 0.95,
    #                 "reasoning": f"Phát hiện từ khóa đặc biệt '{keyword}' trong truy vấn"
    #             }

    #     pc_builder_pattern = r"(xây dựng|build|lắp ráp|cấu hình).*?(pc|máy tính|cấu hình)"
    #     if re.search(pc_builder_pattern, user_query_lower) or "triệu" in user_query_lower:
    #         return {
    #             "intent": "pc_builder",
    #             "confidence": 0.9,
    #             "reasoning": f"Phát hiện yêu cầu xây dựng PC trong truy vấn"
    #         }

    #     order_pattern = r"(mua|đặt|order).*?(hàng|sản phẩm)"
    #     if re.search(order_pattern, user_query_lower):
    #         return {
    #             "intent": "order_processor",
    #             "confidence": 0.9,
    #             "reasoning": f"Phát hiện yêu cầu đặt hàng trong truy vấn"
    #         }

    #     for intent, keywords in self.keywords.items():
    #         for keyword in keywords:
    #             if keyword in user_query_lower:
    #                 return {
    #                     "intent": intent,
    #                     "confidence": 0.85,
    #                     "reasoning": f"Phát hiện từ khóa '{keyword}' thuộc loại {intent}"
    #                 }

    #     return None

    async def classify_intent(self, user_query: str) -> Dict[str, Any]:
        # keyword_result = self._keyword_based_classification(user_query)
        # if keyword_result:
        #     print("intent_result (keyword-based)", keyword_result)
        #     return keyword_result

        try:
            from agents import Runner

            # Format the prompt message
            prompt = f"Phân loại đoạn text này: \"{user_query}\""

            # Use the Runner to execute the agent
            response = await Runner.run(
                self.intent_classifier,
                [{"role": "system", "content": self.intent_classifier.instructions},
                 {"role": "user", "content": prompt}]
            )

            try:
                response_text = response.final_output
                json_text = extract_json_from_response(response_text)

                try:
                    result = json.loads(json_text)

                    if result.get("intent") not in self.agent_types:
                        print(
                            f"Intent '{result.get('intent')}' không nằm trong danh sách agent_types hợp lệ, chuyển sang 'general'")
                        result["intent"] = "general"
                        result["confidence"] = 0.5
                        result["reasoning"] = "Intent không hợp lệ, chuyển sang agent chung"

                    return result
                except json.JSONDecodeError as e:
                    # In ra lỗi và chuỗi JSON để debug
                    print(
                        f"JSON decode error: {e}, extracted JSON: '{json_text}'")
                    print(f"Full response: '{response_text}'")

                    return {
                        "intent": "general",
                        "confidence": 0.5,
                        "reasoning": f"Error parsing JSON response: {e}, defaulting to general agent"
                    }

            except Exception as e:
                print(
                    f"Error processing response: {e}, response: '{response.final_output}'")

                return {
                    "intent": "general",
                    "confidence": 0.5,
                    "reasoning": f"Error processing response: {e}, defaulting to general agent"
                }

        except Exception as e:
            print(f"Intent classification error: {e}")
            return {
                "intent": "general",
                "confidence": 0.5,
                "reasoning": f"Error in classification: {e}, defaulting to general agent"
            }

    async def route_query(self, user_query: str) -> str:
        # user_query_lower = user_query.lower()

        # if any(keyword in user_query_lower for keyword in self.keywords["pc_builder"]):
        #     pc_builder_pattern = r"(xây dựng|cấu hình|build|lắp).*(pc|máy tính)"
        #     if re.search(pc_builder_pattern, user_query_lower) or "triệu" in user_query_lower:
        #         print(
        #             f"Phân loại '{user_query}' là pc_builder dựa trên từ khóa")
        #         return "pc_builder"

        # if any(keyword in user_query_lower for keyword in ["chip", "intel", "core i", "amd", "ryzen", "cpu"]):
        #     print(
        #         f"Phân loại '{user_query}' là product_advisor dựa trên từ khóa CPU/chip")
        #     return "product_advisor"

        # order_pattern = r"(đặt|mua|order).*(hàng|sản phẩm)"
        # if re.search(order_pattern, user_query_lower) or any(keyword in user_query_lower for keyword in ["muốn mua", "mua ngay", "đặt hàng"]):
        #     print(
        #         f"Phân loại '{user_query}' là order_processor dựa trên từ khóa")
        #     return "order_processor"

        # policy_pattern = r"(chính sách|bảo hành|đổi trả|hoàn tiền|thanh toán|giao hàng|vận chuyển)"
        # if re.search(policy_pattern, user_query_lower):
        #     print(
        #         f"Phân loại '{user_query}' là policy_advisor dựa trên từ khóa")
        #     return "policy_advisor"

        # product_pattern = r"(card|ram|mainboard|cpu|gpu|vga|ổ cứng|nguồn|case|tản nhiệt)"
        # if re.search(product_pattern, user_query_lower):
        #     print(
        #         f"Phân loại '{user_query}' là product_advisor dựa trên từ khóa")
        #     return "product_advisor"

        intent_result = await self.classify_intent(user_query)
        print("intent_result", intent_result)

        intent = intent_result.get("intent", "general")
        print("intent", intent)

        confidence = intent_result.get("confidence", 0)
        if confidence < 0.2:
            return "general"

        if intent not in self.agent_types:
            print(
                f"Không tìm thấy agent cho intent: {intent}, chuyển về general")
            return "general"

        return intent
