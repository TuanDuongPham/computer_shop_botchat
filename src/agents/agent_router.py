from typing import Dict, Any, List
from agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from src.services.shared_state import SharedStateService
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

        self.shared_state = SharedStateService()

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

    def set_recently_advised_products(self, products: List[Dict[str, Any]]):
        self.shared_state.set_recently_advised_products(products)
        print(f"AgentRouter: Đã lưu trữ sản phẩm tư vấn gần nhất qua SharedStateService")

    async def classify_intent(self, user_query: str) -> Dict[str, Any]:
        try:
            from agents import Runner

            prompt = f"Phân loại đoạn text này: \"{user_query}\""

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
        order_keywords = ["đặt hàng", "mua ngay", "order", "thanh toán", "mua", "đặt", "lấy",
                          "chốt đơn", "xác nhận", "đồng ý", "ok", "được", "chốt"]
        config_keywords = ["cấu hình", "pc", "máy tính", "bộ máy", "như trên", "vừa rồi",
                           "vừa tư vấn", "bộ này", "cái này", "cái đó", "như vậy", "như thế"]

        recently_advised_products = self.shared_state.get_recently_advised_products()
        recently_advised_pc = self.shared_state.is_recently_advised_pc()

        if len(user_query.split()) <= 5 and any(keyword in user_query.lower() for keyword in order_keywords):
            if recently_advised_products:
                print(
                    "Phát hiện câu đơn giản về đặt hàng, chuyển hướng đến order_processor")
                return "order_processor"

        if (any(keyword in user_query.lower() for keyword in order_keywords) and
                any(keyword in user_query.lower() for keyword in config_keywords)):
            if recently_advised_products:
                print(
                    "Phát hiện ý định đặt hàng cấu hình, chuyển hướng đến order_processor")
                return "order_processor"

        if recently_advised_products:
            try:
                from src.agents.order_processor import OrderProcessorAgent
                order_processor = OrderProcessorAgent()
                result = await order_processor.detect_advised_pc_intent(user_query)
                if len(result) == 5:
                    is_ordering_pc, confidence, reasoning, is_mentioned, mentioned_part = result
                else:
                    is_ordering_pc, confidence, reasoning = result

                if is_ordering_pc and confidence >= 0.7:
                    print(
                        f"LLM phát hiện ý định đặt hàng (độ tin cậy: {confidence}): {reasoning}")
                    return "order_processor"
            except Exception as e:
                print(f"Lỗi khi phát hiện ý định đặt hàng: {e}")

        if any(keyword in user_query.lower() for keyword in order_keywords) and recently_advised_products:
            print("Phát hiện từ khóa đặt hàng khi có sản phẩm tư vấn gần đây")
            return "order_processor"

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
