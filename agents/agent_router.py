from typing import Dict, Any
from openai import OpenAI
from ..config import OPENAI_API_KEY, OPENAI_MODEL
import json


class AgentRouter:
    def __init__(self, config=None):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.config = config or {}
        self.agent_types = {
            "product_advisor": "Tư vấn linh kiện máy tính",
            "policy_advisor": "Tư vấn chính sách cửa hàng",
            "pc_builder": "Tư vấn xây dựng PC",
            "order_processor": "Xác nhận đặt hàng và gửi email",
            "general": "Chào hỏi và hỏi đáp chung"
        }

    def classify_intent(self, user_query: str) -> Dict[str, Any]:
        try:
            prompt = f"""
            Bạn là một AI phân loại ý định của người dùng trong cửa hàng máy tính. 
            Phân loại đoạn text sau vào một trong các danh mục sau:
            
            1. product_advisor: Các câu hỏi về thông tin, tư vấn, so sánh linh kiện máy tính
            2. policy_advisor: Các câu hỏi về chính sách cửa hàng, bảo hành, đổi trả, thanh toán
            3. pc_builder: Yêu cầu xây dựng cấu hình PC hoàn chỉnh theo nhu cầu
            4. order_processor: Yêu cầu đặt hàng, xác nhận đơn hàng, thanh toán
            5. general: Chào hỏi, hỏi đáp chung không liên quan đến 4 loại trên
            
            Đoạn text: "{user_query}"
            
            Hãy trả về kết quả dưới dạng JSON với format:
            {{
                "intent": "<loại agent>",
                "confidence": <điểm tin cậy từ 0.0 đến 1.0>,
                "reasoning": "<giải thích ngắn gọn lý do>"
            }}
            """

            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"Intent classification error: {e}")
            # Default to general category on error
            return {
                "intent": "general",
                "confidence": 0.5,
                "reasoning": "Error in classification, defaulting to general agent"
            }

    def route_query(self, user_query: str) -> str:
        intent_result = self.classify_intent(user_query)
        intent = intent_result.get("intent", "general")
        confidence = intent_result.get("confidence", 0)
        if confidence < 0.6:
            return "general"

        return intent
