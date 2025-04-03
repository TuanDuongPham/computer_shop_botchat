from src.services.policy_search import PolicySearchService
from src.services.vietnamese_llm_helper import VietnameseLLMHelper
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from src.config import OPENAI_MODEL, OPENAI_API_KEY


class PolicyAdvisorAgent:
    def __init__(self):
        self.policy_search = PolicySearchService()
        self.vi_helper = VietnameseLLMHelper()
        self.model_client = OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY)
        )

        # Create agent using OpenAI Agent SDK
        self.agent = Agent(
            name="PolicyAdvisor",
            model=self.model_client,
            handoff_description="Specialist agent for Policy Advisor",
            handoffs=[self.handle_query],
            instructions="""Bạn là chuyên gia về chính sách của cửa hàng TechPlus.
            Nhiệm vụ của bạn là giải đáp các thắc mắc liên quan đến chính sách của cửa hàng, bao gồm:
            - Chính sách bảo hành
            - Chính sách đổi trả và hoàn tiền
            - Chính sách giao hàng
            - Chính sách thanh toán và trả góp
            - Chính sách bảo mật thông tin khách hàng
            - Chương trình khách hàng thân thiết
            - Và các chính sách khác của cửa hàng
            
            Khi trả lời, hãy:
            1. Cung cấp thông tin chính xác, đầy đủ từ chính sách của cửa hàng
            2. Giải thích các điều khoản bằng ngôn ngữ dễ hiểu
            3. Đưa ra các ví dụ cụ thể nếu cần
            
            Bạn có quyền truy cập vào cơ sở dữ liệu chính sách và có thể tìm kiếm thông tin liên quan đến câu hỏi của khách hàng.
            """,
        )

    @function_tool
    async def search_policy(self, query: str, language: str = "vi", n_results: int = 2):
        """Search for relevant policy information based on the query."""
        search_results = self.policy_search.search_policy(
            query, language, n_results)
        return search_results

    @function_tool
    async def format_policy_response(self, search_results):
        """Format the policy search results into a readable response."""
        return self.policy_search.format_policy_response(search_results)

    @function_tool
    async def handle_query(self, query: str, language: str = "vi"):
        """Handle a policy-related query from a user."""
        try:
            # Step 1: Search for policy information based on query
            search_results = await self.search_policy(query, language, n_results=3)

            if not search_results or not search_results.get('results') or not search_results['results'].get('documents'):
                return "Xin lỗi, tôi không tìm thấy thông tin chính sách liên quan đến câu hỏi của bạn. Bạn có thể nêu cụ thể hơn hoặc hỏi về chủ đề khác như 'bảo hành', 'đổi trả', 'thanh toán' không?"

            # Step 2: Format policy information
            formatted_policy = await self.format_policy_response(search_results)

            # Step 3: Combine the policy information with enhanced content
            # Get original query and enhanced query for context
            original_query = search_results.get('original_query', query)
            enhanced_query = search_results.get('enhanced_query', query)

            # Extract top policy sections
            policy_sections = []
            if search_results and search_results.get('results') and search_results['results'].get('metadatas'):
                for metadata in search_results['results']['metadatas'][0]:
                    if 'title' in metadata and 'path' in metadata:
                        section = {
                            'title': metadata.get('title', ''),
                            'path': metadata.get('path', '')
                        }
                        if section not in policy_sections:
                            policy_sections.append(section)

            # Create paths for guidance
            policy_paths = []
            for section in policy_sections:
                if section.get('path'):
                    policy_paths.append(section.get('path'))

            # Step 4: Prepare prompt for the response generation
            prompt = f"""
            Người dùng đang hỏi về chính sách: "{original_query}"
            
            Dựa trên truy vấn, tôi đã tìm thấy các thông tin chính sách sau:
            
            {formatted_policy}
            
            Các phần chính sách liên quan:
            {', '.join(policy_paths) if policy_paths else 'Không có phần cụ thể'}
            
            Hãy trả lời người dùng với thông tin trên, đảm bảo:
            1. Cung cấp thông tin chính xác từ chính sách
            2. Giải thích các điều khoản bằng ngôn ngữ dễ hiểu
            3. Nếu cần, đưa ra ví dụ cụ thể để minh họa
            4. Hỏi xem người dùng cần thêm thông tin về chính sách nào khác không
            
            Định dạng câu trả lời rõ ràng, có cấu trúc dễ đọc.
            """

            # Step 5: Generate response using the agent
            response = await Runner.run(
                self.agent,
                messages=[
                    {"role": "system", "content": self.agent.instructions},
                    {"role": "user", "content": prompt}
                ],
            )

            return response.final_output

        except Exception as e:
            print(f"Error in PolicyAdvisorAgent.handle_query: {e}")
            return f"Xin lỗi, tôi đang gặp sự cố khi tìm kiếm thông tin chính sách. Vui lòng thử lại sau hoặc hỏi về một chính sách cụ thể như 'bảo hành', 'đổi trả' hoặc 'thanh toán'."
