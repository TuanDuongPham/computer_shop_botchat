from typing import Dict, Any
from agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from src.config import OPENAI_API_KEY, OPENAI_MODEL
import json


class AgentRouter:
    def __init__(self, config=None):
        self.config = config or {}
        self.agent_types = {
            "product_advisor": "Tư vấn linh kiện máy tính",
            "policy_advisor": "Tư vấn chính sách cửa hàng",
            "pc_builder": "Tư vấn xây dựng PC",
            "order_processor": "Xác nhận đặt hàng và gửi email",
            "general_advisor": "Chào hỏi và hỏi đáp chung"
        }

        # Initialize OpenAI model client
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
            Phân loại đoạn text vào một trong các danh mục sau:
            
            1. product_advisor: Các câu hỏi về thông tin, tư vấn, so sánh linh kiện máy tính, ao gồm các sản phẩm như CPU, GPU, RAM, Mainboard, SSD, HDD, PSU, Case máy tính
            2. policy_advisor: Các câu hỏi về chính sách cửa hàng, bảo hành, đổi trả, thanh toán
            3. pc_builder: Yêu cầu xây dựng cấu hình PC hoàn chỉnh theo nhu cầu dựa trên các sản phẩm có trong cơ sở dữ liệu của cửa hàng
            4. order_processor: Yêu cầu đặt hàng, xác nhận đơn hàng, thanh toán
            5. general_advisor: Chào hỏi, hỏi đáp chung không liên quan đến 4 loại trên
            
            Trả về kết quả dưới dạng JSON với format:
            {
                "intent": "<loại agent>",
                "confidence": <điểm tin cậy từ 0.0 đến 1.0>,
                "reasoning": "<giải thích ngắn gọn lý do>"
            }
            """,
        )

    async def classify_intent(self, user_query: str) -> Dict[str, Any]:
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

            # Extract the JSON response
            result = json.loads(response.final_output)
            return result

        except Exception as e:
            print(f"Intent classification error: {e}")
            # Default to general category on error
            return {
                "intent": "general",
                "confidence": 0.5,
                "reasoning": "Error in classification, defaulting to general agent"
            }

    async def route_query(self, user_query: str) -> str:
        # If query is very short or seems like a greeting, route directly to general
        if len(user_query.strip()) < 10 or any(greeting in user_query.lower() for greeting in
                                               ["xin chào", "hello", "hi", "chào", "hey", "good morning", "tạm biệt", "cảm ơn"]):
            return "general"

        # Otherwise use intent classification
        intent_result = await self.classify_intent(user_query)
        print("intent_result", intent_result)
        intent = intent_result.get("intent", "general")
        print("intent", intent)
        confidence = intent_result.get("confidence", 0)

        if confidence < 0.5:
            return "general"

        return intent
