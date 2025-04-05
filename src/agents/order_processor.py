from agents import Agent, Runner, FunctionTool, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from src.config import OPENAI_MODEL, OPENAI_API_KEY
from src.services.shared_state import SharedStateService
import re
import json
import time
from datetime import datetime, timedelta
import random
from src.services.price_utils import format_price_usd_to_vnd, convert_usd_to_vnd


class OrderProcessorAgent:
    def __init__(self):
        self.model_client = OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY)
        )

        # Shared state service
        self.shared_state = SharedStateService()

        # Create agent using OpenAI Agent SDK
        self.agent = Agent(
            name="OrderProcessor",
            model=self.model_client,
            handoff_description="Specialist agent for order processing",
            handoffs=[self.extract_product_from_text],
            instructions="""Bạn là chuyên gia xử lý đơn hàng của cửa hàng TechPlus.
            Nhiệm vụ của bạn là hỗ trợ khách hàng hoàn tất quá trình đặt hàng, thu thập thông tin cần thiết, 
            và cung cấp xác nhận đơn hàng.
            
            Khi xử lý đơn hàng, bạn cần:
            1. Thu thập thông tin cá nhân của khách hàng (tên, số điện thoại, địa chỉ giao hàng)
            2. Xác nhận sản phẩm mà khách hàng muốn đặt (loại sản phẩm, số lượng)
            3. Cung cấp thông tin về các phương thức thanh toán
            4. Xác nhận thời gian giao hàng dự kiến
            5. Tạo mã đơn hàng và gửi thông tin tóm tắt
            
            Thông tin về phương thức thanh toán của TechPlus:
            - Thanh toán tiền mặt khi nhận hàng (COD)
            - Chuyển khoản ngân hàng (BIDV, Vietcombank, Techcombank)
            - Thanh toán qua ví điện tử (MoMo, ZaloPay, VNPay)
            - Thanh toán thẻ tín dụng/ghi nợ
            - Trả góp qua các đối tác tài chính (Home Credit, FE Credit)
            
            Thông tin về thời gian giao hàng:
            - Nội thành: 2-24 giờ (tùy sản phẩm)
            - Ngoại thành: 24-48 giờ
            - Tỉnh thành khác: 2-5 ngày làm việc
            
            Hãy luôn giữ thái độ chuyên nghiệp, lịch sự và hướng dẫn khách hàng qua từng bước của quá trình đặt hàng.
            """,
        )

        # Agent riêng để phân tích ý định và kiểm tra cấu hình đã tư vấn
        self.intent_analyzer = Agent(
            name="OrderIntentAnalyzer",
            model=self.model_client,
            instructions="""Bạn là một chuyên gia phân tích ý định đặt hàng. 
            Nhiệm vụ của bạn là xác định xem người dùng có đang muốn mua cấu hình PC vừa được tư vấn không.
            
            Hãy phân tích xem trong tin nhắn của người dùng có các dấu hiệu sau không:
            1. Đề cập đến "cấu hình này", "cấu hình trên", "bộ PC này", "máy tính này"
            2. Sử dụng các từ ngữ đặt hàng ("mua", "đặt", "lấy", "order", "checkout")
            3. Đề cập đến việc thực hiện giao dịch mua bán cấu hình PC vừa được tư vấn
            
            Trả về dưới dạng JSON với format:
            {
                "is_ordering_advised_pc": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "Giải thích lý do"
            }
            """
        )

        self.orders = {}

        # Payment methods
        self.payment_methods = [
            "Thanh toán tiền mặt khi nhận hàng (COD)",
            "Chuyển khoản ngân hàng",
            "Thanh toán qua ví điện tử (MoMo, ZaloPay, VNPay)",
            "Thanh toán thẻ tín dụng/ghi nợ",
            "Trả góp qua đối tác tài chính"
        ]

        # Dict of payment details
        self.payment_details = {
            "Chuyển khoản ngân hàng": {
                "BIDV": "TechPlus JSC - 21010000123456 - Chi nhánh Hà Nội",
                "Vietcombank": "TechPlus JSC - 0011000123456 - Chi nhánh HCM",
                "Techcombank": "TechPlus JSC - 19033123456789 - Chi nhánh Đà Nẵng"
            },
            "Thanh toán qua ví điện tử (MoMo, ZaloPay, VNPay)": {
                "MoMo": "TechPlus - 0909123456",
                "ZaloPay": "TechPlus - 0909123456",
                "VNPay": "Quét mã QR trên app ngân hàng"
            }
        }

    async def detect_advised_pc_intent(self, text):
        """Phát hiện ý định mua cấu hình PC vừa được tư vấn."""
        try:
            # Nếu chưa từng tư vấn cấu hình PC, trả về False ngay
            recently_advised_pc = self.shared_state.is_recently_advised_pc()
            recently_advised_products = self.shared_state.get_recently_advised_products()

            if not recently_advised_pc and not recently_advised_products:
                return False, 0.0, "Chưa có tư vấn cấu hình PC trước đó"

            # Các từ khoá đặc trưng cho việc muốn mua cấu hình vừa tư vấn - mở rộng danh sách
            pc_keywords = [
                "cấu hình này", "cấu hình trên", "bộ pc này", "pc này", "máy tính này",
                "máy này", "hệ thống này", "cấu hình vừa tư vấn", "bộ máy vừa tư vấn",
                "bộ máy này", "cấu hình vừa rồi", "như trên", "như vậy", "như thế",
                "pc đó", "bộ đó", "cái này", "vừa tư vấn", "vừa giới thiệu",
                "bộ này", "mua bộ này", "cấu hình đó", "mua cái này", "mua bộ đó",
                "mua hết", "lấy hết", "lấy tất cả", "lấy full", "đặt full bộ",
                "đặt toàn bộ", "mua nguyên bộ", "lấy nguyên cấu hình", "mua đúng vậy",
                "toàn bộ cấu hình", "cả cấu hình"
            ]

            # Các động từ mua sắm - mở rộng danh sách
            purchase_verbs = [
                "mua", "đặt", "lấy", "order", "checkout", "thanh toán", "giao",
                "xuống đơn", "chốt", "lấy luôn", "đặt mua", "đặt hàng", "mua ngay",
                "đặt ngay", "xác nhận", "đồng ý", "ok", "ổn", "được", "tôi lấy",
                "tôi mua", "cho tôi", "cho mình", "tôi muốn", "mình muốn"
            ]

            # Kiểm tra nhanh bằng regex
            text_lower = text.lower()

            # Trường hợp đặc biệt: nếu người dùng chỉ nói "đặt hàng", "mua ngay", "ok"
            # và đang có sản phẩm vừa được tư vấn
            simple_purchase_commands = [
                "đặt hàng", "mua ngay", "ok", "đặt mua", "mua luôn", "chốt", "lấy"]
            if recently_advised_pc and any(cmd in text_lower for cmd in simple_purchase_commands) and len(text_lower.split()) <= 5:
                return True, 0.98, "Lệnh đặt hàng đơn giản sau khi tư vấn cấu hình PC"

            # Nếu có từ khoá PC + động từ mua → đây là ý định mua cấu hình đã tư vấn
            if any(kw in text_lower for kw in pc_keywords) and any(verb in text_lower for verb in purchase_verbs):
                return True, 0.95, "Phát hiện từ khoá mua cấu hình PC vừa tư vấn"

            # Kiểm tra các trường hợp người dùng chỉ viết rất ngắn gọn
            short_text = len(text_lower.split()) <= 10
            has_pc_term = "pc" in text_lower or "cấu hình" in text_lower or "máy tính" in text_lower or "bộ máy" in text_lower
            has_purchase_term = any(
                verb in text_lower for verb in purchase_verbs)

            if short_text and has_pc_term and has_purchase_term and recently_advised_pc:
                return True, 0.90, "Phát hiện câu ngắn gọn có ý định mua cấu hình PC"

            # Phân tích chi tiết hơn bằng LLM
            prompt = f"""
            Phân tích xem người dùng có đang muốn mua cấu hình PC vừa được tư vấn không.
            
            Tin nhắn của người dùng: "{text}"
            
            Thông tin bổ sung:
            - Người dùng vừa được tư vấn một cấu hình PC đầy đủ
            - Các thành phần trong cấu hình gồm: {", ".join([p.get("name", "") for p in recently_advised_products if p.get("name")])}
            
            Nếu dựa vào tin nhắn của người dùng, họ có vẻ như muốn mua cấu hình PC vừa được tư vấn, hãy trả về JSON với is_ordering_advised_pc=true.
            Đặc biệt chú ý các yếu tố sau:
            1. Người dùng có thể chỉ viết rất ngắn gọn như "đặt hàng", "mua", "lấy", "ok"
            2. Người dùng có thể đề cập đến "cấu hình này", "như trên", "PC đó" hoặc các từ tương tự
            3. Xét cả ngữ cảnh người đã được tư vấn cấu hình PC và đang trong quá trình mua sắm
            """

            response = await Runner.run(
                self.intent_analyzer,
                [{"role": "user", "content": prompt}]
            )

            # Trích xuất JSON từ phản hồi
            response_text = response.final_output

            # Tìm và trích xuất phần JSON từ văn bản
            import re
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)

            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    is_ordering = result.get("is_ordering_advised_pc", False)
                    confidence = result.get("confidence", 0.0)
                    reasoning = result.get("reasoning", "")

                    return is_ordering, confidence, reasoning
                except json.JSONDecodeError:
                    pass

            # Fallback nếu không thể phân tích được JSON
            # Kiểm tra thêm một lần nữa với heuristic đơn giản
            if recently_advised_pc:
                # Nếu tin nhắn ngắn và có từ liên quan đến mua hàng
                if len(text_lower.split()) <= 5 and any(verb in text_lower for verb in purchase_verbs):
                    return True, 0.85, "Tin nhắn ngắn có từ khóa mua hàng sau khi tư vấn PC"

            return False, 0.0, "Không thể phân tích được ý định"

        except Exception as e:
            print(f"Lỗi khi phát hiện ý định mua cấu hình PC: {e}")
            return False, 0.0, f"Lỗi: {str(e)}"

    async def extract_product_from_text(self, text):
        """Sử dụng LLM để trích xuất tên sản phẩm từ văn bản."""
        try:
            # Kiểm tra trước xem có phải đang muốn mua cấu hình vừa tư vấn không
            is_ordering_pc, confidence, reasoning = await self.detect_advised_pc_intent(text)

            recently_advised_products = self.shared_state.get_recently_advised_products()
            recently_advised_pc = self.shared_state.is_recently_advised_pc()

            if is_ordering_pc and confidence >= 0.7:
                print(
                    f"Phát hiện ý định mua cấu hình PC vừa tư vấn: {reasoning} (độ tin cậy: {confidence})")
                return recently_advised_products

            # Kiểm tra nhanh các cụm từ đơn giản về đặt hàng
            simple_order_terms = ["đặt hàng", "mua ngay",
                                  "mua luôn", "đặt mua", "đơn hàng", "chốt đơn"]
            if any(term in text.lower() for term in simple_order_terms) and len(text.split()) <= 5:
                # Nếu là câu đơn giản và có sản phẩm đã tư vấn gần đây
                if recently_advised_products:
                    return recently_advised_products

            # Chuẩn bị danh sách sản phẩm đã tư vấn gần đây cho context
            recent_products_text = ""
            if recently_advised_products:
                recent_products_text = "Sản phẩm đã tư vấn gần đây:\n"
                for product in recently_advised_products:
                    product_name = product.get('name', '')
                    category = product.get('category', '')
                    quantity = product.get('quantity', 1)
                    if product_name:
                        if category:
                            recent_products_text += f"- {product_name} (Loại: {category}) x{quantity}\n"
                        else:
                            recent_products_text += f"- {product_name} x{quantity}\n"

            # Kiểm tra nếu có từ khóa cấu hình nhưng không nói rõ là cái nào
            pc_keywords = ["cấu hình", "pc", "máy tính",
                           "bộ máy", "hệ thống", "bộ này", "như trên"]
            if any(keyword in text.lower() for keyword in pc_keywords) and recently_advised_pc:
                return recently_advised_products

            prompt = f"""
            Trích xuất tên sản phẩm máy tính hoặc linh kiện từ đoạn văn bản sau:
            "{text}"
            
            {recent_products_text}
            
            Hãy trả về một danh sách JSON các sản phẩm được đề cập, với mỗi sản phẩm gồm tên và số lượng.
            Ví dụ: [
                {{"name": "CPU Intel Core i7-13700K", "quantity": 1}},
                {{"name": "RAM Kingston Fury 32GB", "quantity": 2}}
            ]
            
            Lưu ý:
            - Nếu văn bản chỉ thể hiện ý định mua hàng chung chung (như "Tôi muốn đặt hàng", "Đặt hàng ngay", "Mua sản phẩm", "OK" hoặc "Đồng ý")
              và có sản phẩm đã tư vấn gần đây, hãy sử dụng thông tin từ sản phẩm đã tư vấn.
            
            - Nếu văn bản có đề cập đến "cấu hình này", "cấu hình trên", "bộ PC này", "như trên", "vậy", "đó", và các sản phẩm tư vấn gồm nhiều linh kiện
              máy tính, hãy hiểu rằng người dùng muốn mua toàn bộ cấu hình vừa được tư vấn.
            
            - Nếu văn bản có đề cập đến loại sản phẩm (như "CPU", "RAM", "card đồ họa") nhưng không nêu cụ thể tên,
              và có sản phẩm tương ứng đã tư vấn gần đây, hãy sử dụng thông tin từ sản phẩm đã tư vấn với loại đó.
            
            - Nếu người dùng sử dụng các từ như "ok", "đồng ý", "được", "chốt", "mua đi", khi vừa có tư vấn sản phẩm, 
              hãy hiểu đó là họ muốn mua sản phẩm đã tư vấn.
            
            - Nếu không có sản phẩm cụ thể nào được đề cập, hãy trả về danh sách trống [].
            
            Chỉ trả về đối tượng JSON, không cần thêm giải thích.
            """

            response = await Runner.run(
                Agent(
                    name="ProductExtractor",
                    model=self.model_client,
                    instructions="Trích xuất tên sản phẩm từ văn bản và dữ liệu sản phẩm đã tư vấn"
                ),
                [{"role": "user", "content": prompt}]
            )

            try:
                # Trích xuất JSON từ phản hồi
                raw_text = response.final_output

                # Tìm dấu [] trong văn bản
                start_idx = raw_text.find('[')
                end_idx = raw_text.rfind(']') + 1

                if start_idx == -1 or end_idx == 0:
                    # Kiểm tra các trường hợp đặc biệt
                    confirmation_keywords = [
                        "ok", "đồng ý", "được", "chốt", "mua", "đặt", "lấy"]
                    if any(keyword in text.lower() for keyword in confirmation_keywords) and len(text.split()) <= 3:
                        if recently_advised_products:
                            return recently_advised_products

                    # Kiểm tra xem có đang muốn mua cấu hình vừa được tư vấn không
                    if any(kw in text.lower() for kw in pc_keywords) and recently_advised_pc:
                        return recently_advised_products

                    return []

                json_text = raw_text[start_idx:end_idx]
                products = json.loads(json_text)

                # Nếu không tìm thấy sản phẩm trong văn bản nhưng có ý định mua hàng
                if not products and recently_advised_products:
                    order_keywords = ["mua", "đặt", "order",
                                      "thanh toán", "lấy", "chốt", "đồng ý", "ok"]
                    if any(keyword in text.lower() for keyword in order_keywords):
                        # Kiểm tra xem có đang muốn mua cấu hình vừa được tư vấn không
                        if any(keyword in text.lower() for keyword in pc_keywords):
                            if recently_advised_pc:
                                return recently_advised_products
                        elif len(text.split()) <= 5:  # Câu ngắn gọn như "OK", "Đồng ý", "Đặt hàng"
                            return recently_advised_products

                return products

            except Exception as e:
                print(f"Lỗi khi phân tích JSON: {e}")
                # Trường hợp lỗi phân tích JSON nhưng có sản phẩm đã tư vấn gần đây
                if recently_advised_products:
                    # Kiểm tra cụm từ đơn giản
                    simple_phrases = ["ok", "đồng ý",
                                      "được", "chốt", "mua", "đặt", "lấy"]
                    if any(phrase in text.lower() for phrase in simple_phrases) and len(text.split()) <= 5:
                        return recently_advised_products

                    # Kiểm tra xem có đang muốn mua cấu hình vừa được tư vấn không
                    pc_keywords = ["cấu hình", "pc",
                                   "máy tính", "bộ máy", "như trên"]
                    if any(keyword in text.lower() for keyword in pc_keywords):
                        if recently_advised_pc:
                            return recently_advised_products
                return []

        except Exception as e:
            print(f"Lỗi khi trích xuất sản phẩm: {e}")
            recently_advised_products = self.shared_state.get_recently_advised_products()
            recently_advised_pc = self.shared_state.is_recently_advised_pc()

            # Trường hợp lỗi chung nhưng có sản phẩm đã tư vấn gần đây
            if recently_advised_products:
                # Kiểm tra cụm từ đơn giản
                if len(text.split()) <= 5:
                    return recently_advised_products

                # Kiểm tra xem có đang muốn mua cấu hình vừa được tư vấn không
                pc_keywords = ["cấu hình", "pc",
                               "máy tính", "bộ máy", "như trên"]
                if any(keyword in text.lower() for keyword in pc_keywords):
                    if recently_advised_pc:
                        return recently_advised_products
            return []

    async def handle_query(self, query: str, language: str = "vi"):
        try:
            # Lấy dữ liệu từ shared state
            recently_advised_products = self.shared_state.get_recently_advised_products()
            recently_advised_pc = self.shared_state.is_recently_advised_pc()

            # Kiểm tra xem có phải đang muốn mua cấu hình vừa tư vấn không
            is_ordering_pc, confidence, reasoning = await self.detect_advised_pc_intent(query)

            print(
                f"Phát hiện ý định đặt cấu hình: {is_ordering_pc}, độ tin cậy: {confidence}, lý do: {reasoning}")

            # Nếu đây là ý định mua cấu hình vừa tư vấn (với độ tin cậy cao)
            if is_ordering_pc and confidence >= 0.7:
                # Kiểm tra xem có sản phẩm đã tư vấn không
                if not recently_advised_products:
                    return "Xin lỗi, tôi không tìm thấy thông tin về cấu hình PC vừa tư vấn. Vui lòng cho biết bạn muốn đặt mua sản phẩm nào?"

                product_info = ""
                for product in recently_advised_products:
                    name = product.get('name', '')
                    quantity = product.get('quantity', 1)
                    price = product.get('price', 0)

                    if name:
                        # Định dạng giá nếu có
                        price_display = ""
                        if price:
                            from src.services.price_utils import format_price_usd_to_vnd
                            price_display = f" - {format_price_usd_to_vnd(price)}"

                        product_info += f"- {name} x{quantity}{price_display}\n"

                # Tính tổng tiền
                from src.services.price_utils import format_price_usd_to_vnd
                total_price = sum([p.get('price', 0) * p.get('quantity', 1)
                                  for p in recently_advised_products])
                total_price_formatted = format_price_usd_to_vnd(total_price)

                response_with_signal = {
                    "content": f"Bạn muốn đặt hàng cấu hình PC vừa được tư vấn:\n{product_info}\nTổng giá trị: {total_price_formatted}\n\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                    "show_order_form": True,
                    "products": recently_advised_products
                }
                return response_with_signal

            # Tiếp tục xử lý các trường hợp khác
            # Trích xuất sản phẩm từ văn bản đầu vào
            extracted_products = await self.extract_product_from_text(query)

            # Nếu không trích xuất được sản phẩm từ đầu vào
            if not extracted_products:
                # Kiểm tra nếu có các từ khóa liên quan đến đặt hàng
                order_keywords = ["đặt hàng", "mua ngay", "order", "thanh toán",
                                  "mua", "đặt", "lấy", "chốt đơn", "xác nhận"]
                config_keywords = ["cấu hình", "pc", "máy tính", "bộ máy",
                                   "hệ thống", "bộ này", "như trên"]

                has_order_intent = any(keyword in query.lower()
                                       for keyword in order_keywords)
                has_config_intent = any(keyword in query.lower()
                                        for keyword in config_keywords)

                # Xử lý khi có ý định đặt hàng nhưng không xác định được sản phẩm cụ thể
                if has_order_intent:
                    # Ưu tiên kiểm tra cấu hình PC được tư vấn gần nhất
                    if (has_config_intent or len(query.split()) <= 5) and recently_advised_products:
                        # Sử dụng cấu hình PC được tư vấn gần nhất
                        products = recently_advised_products
                        product_info = ""
                        for product in products:
                            name = product.get('name', '')
                            quantity = product.get('quantity', 1)
                            price = product.get('price', 0)

                            if name:
                                # Định dạng giá nếu có
                                price_display = ""
                                if price:
                                    from src.services.price_utils import format_price_usd_to_vnd
                                    price_display = f" - {format_price_usd_to_vnd(price)}"

                                product_info += f"- {name} x{quantity}{price_display}\n"

                        # Tính tổng tiền
                        from src.services.price_utils import format_price_usd_to_vnd
                        total_price = sum(
                            [p.get('price', 0) * p.get('quantity', 1) for p in products])
                        total_price_formatted = format_price_usd_to_vnd(
                            total_price)

                        response_with_signal = {
                            "content": f"Bạn muốn đặt hàng sản phẩm đã được tư vấn:\n{product_info}\nTổng giá trị: {total_price_formatted}\n\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                            "show_order_form": True,
                            "products": products
                        }
                        return response_with_signal
                    else:
                        return "Bạn muốn đặt sản phẩm gì? Vui lòng cung cấp thêm thông tin về sản phẩm bạn muốn mua, hoặc cho biết bạn muốn mua cấu hình vừa tư vấn không?"
                else:
                    return "Bạn muốn đặt sản phẩm gì? Vui lòng cung cấp thêm thông tin về sản phẩm bạn muốn mua."
            else:
                # Nếu trích xuất được sản phẩm từ đầu vào
                product_info = ""
                for product in extracted_products:
                    quantity = product.get("quantity", 1)
                    product_info += f"- {product['name']} x{quantity}\n"

                    # Kiểm tra sản phẩm trong CSDL (đoạn này sẽ cần triển khai thực tế)
                    product_exists = self._check_product_in_database(
                        product['name'])
                    if not product_exists:
                        return f"Xin lỗi, sản phẩm '{product['name']}' hiện không có trong cửa hàng. Vui lòng chọn sản phẩm khác hoặc liên hệ với nhân viên tư vấn."

                # Trường hợp khách hàng gọi tên sản phẩm cụ thể
                response_with_signal = {
                    "content": f"Bạn đã chọn sản phẩm:\n{product_info}\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                    "show_order_form": True,
                    "products": extracted_products
                }

                return response_with_signal

        except Exception as e:
            print(f"Error in OrderProcessorAgent.handle_query: {e}")
            return f"Xin lỗi, tôi đang gặp sự cố khi xử lý đơn đặt hàng. Vui lòng thử lại sau hoặc liên hệ trực tiếp với cửa hàng qua hotline 1900-TECHPLUS."

    def create_order_from_form(self, customer_info, products):
        """Tạo đơn hàng từ thông tin form."""
        try:
            # Tạo đơn hàng trong hệ thống
            order = self.create_order(customer_info, products)

            # Chuẩn bị thông tin thanh toán
            payment_options = "\n".join(
                [f"- {method}" for method in self.payment_methods])

            # Định dạng thông tin sản phẩm
            product_details = ""
            total_price = 0

            for product in order["products"]:
                price = self.format_price(product["price"])
                product_details += f"- {product['name']} x{product['quantity']}: {price}\n"
                total_price += product["price"] * product["quantity"]

            # Tạo phản hồi xác nhận đơn hàng
            confirmation = f"""### Xác nhận đơn hàng #{order['order_id']}

            **Thông tin khách hàng:**
            - Họ tên: {customer_info['customer_name']}
            - Số điện thoại: {customer_info['customer_phone']}
            - Địa chỉ giao hàng: {customer_info['customer_address']}

            **Sản phẩm đã đặt:**
            {product_details}
            **Tổng tiền:** {self.format_price(total_price)}

            **Thời gian giao hàng dự kiến:** {order['delivery_date']} ({order['delivery_time']})

            **Phương thức thanh toán:**
            {payment_options}

            Cảm ơn bạn đã mua hàng tại TechPlus! Chúng tôi sẽ liên hệ với bạn để xác nhận đơn hàng trong thời gian sớm nhất.
            """
            return {"confirmation": confirmation}

        except Exception as e:
            print(f"Error processing order: {e}")
            error_message = f"Xin lỗi, đã xảy ra lỗi khi xử lý đơn hàng: {str(e)}. Vui lòng thử lại hoặc liên hệ với chúng tôi qua số hotline 1900-TECHPLUS."
            return {"confirmation": error_message}
