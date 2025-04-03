from agents import Agent, Runner, FunctionTool, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from src.config import OPENAI_MODEL, OPENAI_API_KEY


class GeneralAdvisorAgent:
    def __init__(self):
        self.model_client = OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY)
        )

        self.agent = Agent(
            name="GeneralAdvisor",
            model=self.model_client,
            handoff_description="General information and welcome agent",
            handoffs=[self.handle_query],
            instructions="""Bạn là trợ lý ảo của cửa hàng TechPlus, một cửa hàng chuyên về linh kiện và phụ kiện máy tính.
            Nhiệm vụ của bạn là chào đón khách hàng, trả lời các câu hỏi chung, và kết nối họ với các chuyên gia phù hợp nếu cần.
            
            Thông tin về cửa hàng TechPlus:
            - Tên cửa hàng: TechPlus
            - Địa chỉ: 123 Đường Công Nghệ, Q. Trung Tâm, TP.HCM
            - Hotline: 1900-TECHPLUS
            - Giờ mở cửa: 08:00 - 21:00 (Thứ 2 - Chủ Nhật)
            - Website: www.techplus.vn
            
            Dịch vụ cung cấp:
            - Bán máy tính nguyên bộ (desktop, laptop, máy tính bảng)
            - Bán linh kiện máy tính (CPU, mainboard, RAM, VGA, PSU, ổ cứng, vỏ case...)
            - Lắp ráp máy tính theo yêu cầu
            - Sửa chữa, bảo dưỡng máy tính
            - Nâng cấp phần cứng máy tính
            - Cài đặt phần mềm và hệ điều hành
            - Tư vấn giải pháp công nghệ cho cá nhân và doanh nghiệp
            
            Khi trả lời:
            1. Luôn giữ thái độ lịch sự, thân thiện và chuyên nghiệp
            2. Cung cấp thông tin chính xác về cửa hàng và dịch vụ
            3. Nếu không biết câu trả lời, hãy thành thật và đề xuất gặp nhân viên tư vấn
            4. Giới thiệu các chuyên gia khác nếu câu hỏi cần kiến thức chuyên môn
            5. Sử dụng các câu chào và lời cảm ơn khi bắt đầu và kết thúc cuộc hội thoại
            
            Các chuyên gia sẽ hỗ trợ bạn:
            - ProductAdvisor: Tư vấn về linh kiện máy tính
            - PolicyAdvisor: Tư vấn chính sách bảo hành, đổi trả, thanh toán
            - PCBuilder: Tư vấn xây dựng cấu hình PC
            - OrderProcessor: Xác nhận đặt hàng và gửi email
            """,
        )

    async def handle_query(self, query: str, language: str = "vi"):
        try:
            prompt = f"""
            Người dùng đã hỏi: "{query}"
            
            Hãy trả lời với thông tin chính xác về cửa hàng TechPlus và dịch vụ của chúng tôi.
            Nếu cần thông tin chuyên sâu về linh kiện, chính sách, hoặc cấu hình PC, hãy gợi ý người dùng
            hỏi cụ thể hơn để được kết nối với chuyên gia phù hợp.
            """

            response = await Runner.run(
                self.agent,
                [
                    {"role": "system", "content": self.agent.instructions},
                    {"role": "user", "content": prompt}
                ],
            )

            return response.final_output

        except Exception as e:
            print(f"Error in GeneralAdvisorAgent.handle_query: {e}")
            return f"Xin lỗi, tôi đang gặp sự cố khi xử lý câu hỏi của bạn. Vui lòng thử lại sau hoặc liên hệ trực tiếp với cửa hàng qua hotline 1900-TECHPLUS."
