from agents import Agent, Runner, FunctionTool, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from src.config import OPENAI_MODEL, OPENAI_API_KEY
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

        # Create agent using OpenAI Agent SDK
        self.agent = Agent(
            name="OrderProcessor",
            model=self.model_client,
            handoff_description="Specialist agent for order processing",
            handoffs=[self.format_price, self.create_order,
                      self.extract_product_from_text],
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
        # Lưu trữ sản phẩm được tư vấn gần nhất
        self.recently_advised_products = []
        # Lưu trữ trạng thái đã tư vấn PC hay chưa
        self.recently_advised_pc = False

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

    def set_recently_advised_products(self, products):
        """Lưu trữ sản phẩm được tư vấn gần nhất."""
        self.recently_advised_products = products
        self.recently_advised_pc = False

        # Kiểm tra nếu danh sách sản phẩm có đủ thành phần của PC
        pc_components = ["CPU", "Motherboard", "RAM", "GPU", "Storage", "PSU"]
        found_components = [product.get("category")
                            for product in products if "category" in product]

        # Nếu có ít nhất 4/6 thành phần chính, đánh dấu là vừa tư vấn PC
        if len(set(found_components).intersection(pc_components)) >= 4:
            self.recently_advised_pc = True
            print("Đã lưu trữ cấu hình PC vừa tư vấn")

        print(f"Đã lưu trữ {len(products)} sản phẩm tư vấn gần nhất")

    async def detect_advised_pc_intent(self, text):
        """Phát hiện ý định mua cấu hình PC vừa được tư vấn."""
        try:
            # Nếu chưa từng tư vấn cấu hình PC, trả về False ngay
            if not self.recently_advised_pc:
                return False, 0.0, "Chưa có tư vấn cấu hình PC trước đó"

            # Các từ khoá đặc trưng cho việc muốn mua cấu hình vừa tư vấn
            pc_keywords = [
                "cấu hình này", "cấu hình trên", "bộ pc này", "pc này",
                "máy tính này", "máy này", "hệ thống này", "cấu hình vừa tư vấn",
                "bộ máy vừa tư vấn", "bộ máy này", "cấu hình vừa rồi", "như trên",
                "như vậy", "như thế", "pc đó", "bộ đó", "cái này"
            ]

            # Các động từ mua sắm
            purchase_verbs = [
                "mua", "đặt", "lấy", "order", "checkout", "thanh toán", "giao",
                "xuống đơn", "chốt", "lấy luôn"
            ]

            # Kiểm tra nhanh bằng regex
            text_lower = text.lower()

            # Nếu có từ khoá PC + động từ mua → đây là ý định mua cấu hình đã tư vấn
            if any(kw in text_lower for kw in pc_keywords) and any(verb in text_lower for verb in purchase_verbs):
                return True, 0.95, "Phát hiện từ khoá mua cấu hình PC vừa tư vấn"

            # Phân tích chi tiết hơn bằng LLM
            prompt = f"""
            Phân tích xem người dùng có đang muốn mua cấu hình PC vừa được tư vấn không.
            
            Tin nhắn của người dùng: "{text}"
            
            Thông tin bổ sung:
            - Người dùng vừa được tư vấn một cấu hình PC đầy đủ
            - Các thành phần trong cấu hình gồm: {", ".join([p.get("name", "") for p in self.recently_advised_products if p.get("name")])}
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
            return False, 0.0, "Không thể phân tích được ý định"

        except Exception as e:
            print(f"Lỗi khi phát hiện ý định mua cấu hình PC: {e}")
            return False, 0.0, f"Lỗi: {str(e)}"

    async def extract_product_from_text(self, text):
        """Sử dụng LLM để trích xuất tên sản phẩm từ văn bản."""
        try:
            # Kiểm tra trước xem có phải đang muốn mua cấu hình vừa tư vấn không
            is_ordering_pc, confidence, reasoning = await self.detect_advised_pc_intent(text)

            if is_ordering_pc and confidence >= 0.7:
                print(
                    f"Phát hiện ý định mua cấu hình PC vừa tư vấn: {reasoning} (độ tin cậy: {confidence})")
                return self.recently_advised_products

            # Chuẩn bị danh sách sản phẩm đã tư vấn gần đây cho context
            recent_products_text = ""
            if self.recently_advised_products:
                recent_products_text = "Sản phẩm đã tư vấn gần đây:\n"
                for product in self.recently_advised_products:
                    product_name = product.get('name', '')
                    category = product.get('category', '')
                    if product_name:
                        if category:
                            recent_products_text += f"- {product_name} (Loại: {category})\n"
                        else:
                            recent_products_text += f"- {product_name}\n"

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
            - Nếu văn bản chỉ thể hiện ý định mua hàng chung chung (như "Tôi muốn đặt hàng", "Đặt hàng ngay", "Mua sản phẩm")
              và có sản phẩm đã tư vấn gần đây, hãy sử dụng thông tin từ sản phẩm đã tư vấn.
            - Nếu văn bản có đề cập đến "cấu hình này", "cấu hình trên", "bộ PC này" và các sản phẩm tư vấn gồm nhiều linh kiện
              máy tính, hãy hiểu rằng người dùng muốn mua toàn bộ cấu hình vừa được tư vấn.
            - Nếu văn bản có đề cập đến loại sản phẩm (như "CPU", "RAM", "card đồ họa") nhưng không nêu cụ thể tên,
              và có sản phẩm tương ứng đã tư vấn gần đây, hãy sử dụng thông tin từ sản phẩm đã tư vấn với loại đó.
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
                    # Kiểm tra xem có đang muốn mua cấu hình vừa được tư vấn không
                    if any(keyword in text.lower() for keyword in ["mua", "đặt", "order", "lấy"]) and any(keyword in text.lower() for keyword in ["cấu hình", "pc", "máy tính", "bộ máy"]):
                        if self.recently_advised_pc:
                            return self.recently_advised_products
                    return []

                json_text = raw_text[start_idx:end_idx]
                products = json.loads(json_text)

                # Nếu không tìm thấy sản phẩm trong văn bản nhưng có ý định mua hàng
                if not products and self.recently_advised_products:
                    if any(keyword in text.lower() for keyword in ["mua", "đặt", "order", "thanh toán", "lấy"]):
                        # Kiểm tra xem có đang muốn mua cấu hình vừa được tư vấn không
                        if any(keyword in text.lower() for keyword in ["cấu hình", "pc", "máy tính", "bộ máy"]):
                            if self.recently_advised_pc:
                                return self.recently_advised_products

                return products
            except Exception as e:
                print(f"Lỗi khi phân tích JSON: {e}")
                # Trường hợp lỗi phân tích JSON nhưng có sản phẩm đã tư vấn gần đây
                if self.recently_advised_products:
                    # Kiểm tra xem có đang muốn mua cấu hình vừa được tư vấn không
                    if any(keyword in text.lower() for keyword in ["mua", "đặt", "order", "lấy"]) and any(keyword in text.lower() for keyword in ["cấu hình", "pc", "máy tính", "bộ máy"]):
                        if self.recently_advised_pc:
                            return self.recently_advised_products
                return []
        except Exception as e:
            print(f"Lỗi khi trích xuất sản phẩm: {e}")
            # Trường hợp lỗi chung nhưng có sản phẩm đã tư vấn gần đây
            if self.recently_advised_products:
                # Kiểm tra xem có đang muốn mua cấu hình vừa được tư vấn không
                if any(keyword in text.lower() for keyword in ["mua", "đặt", "order", "lấy"]) and any(keyword in text.lower() for keyword in ["cấu hình", "pc", "máy tính", "bộ máy"]):
                    if self.recently_advised_pc:
                        return self.recently_advised_products
            return []

    def _extract_customer_info(self, query):
        info = {}

        name_patterns = [
            r'[t|T]ên\s+(?:là|:)?\s+([A-Za-z\sÀ-ỹ]+)',
            r'[t|T]ôi\s+(?:là|tên)?\s+([A-Za-z\sÀ-ỹ]+)'
        ]

        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                info['name'] = match.group(1).strip()
                break

        phone_patterns = [
            r'số\s+(?:điện thoại|ĐT|đt|phone|di động)(?:\s+là|:)?\s+(0\d{9,10})',
            r'(0\d{9,10})'
        ]

        for pattern in phone_patterns:
            match = re.search(pattern, query)
            if match:
                info['phone'] = match.group(1).strip()
                break

        address_patterns = [
            r'địa\s+chỉ(?:\s+là|:)?\s+([^\.]+)',
            r'giao\s+(?:hàng\s+)?(?:đến|tới|tại)(?:\s+địa\s+chỉ)?\s+([^\.]+)'
        ]

        for pattern in address_patterns:
            match = re.search(pattern, query)
            if match:
                info['address'] = match.group(1).strip()
                break

        return info

    def _check_product_in_database(self, product_name):
        """Kiểm tra sản phẩm trong CSDL Postgres."""
        # Đây là hàm mô phỏng, trong thực tế sẽ kết nối với Postgres để tìm kiếm sản phẩm
        # Trả về True nếu tìm thấy, False nếu không tìm thấy
        # TODO: Implement actual database lookup
        return True

    def _generate_order_id(self):
        timestamp = int(time.time())
        random_num = random.randint(1000, 9999)
        return f"TP{timestamp}{random_num}"

    def _estimate_delivery_time(self, address):
        address_lower = address.lower() if address else ""

        inner_city_keywords = ["quận", "hcm", "hồ chí minh",
                               "tphcm", "hà nội", "hanoi", "đà nẵng", "danang"]
        if any(keyword in address_lower for keyword in inner_city_keywords):
            delivery_hours = random.randint(4, 24)
            delivery_date = datetime.now() + timedelta(hours=delivery_hours)
            return delivery_date.strftime("%d/%m/%Y"), "trong vòng 24 giờ"

        suburban_keywords = ["huyện", "thị xã", "bình dương",
                             "đồng nai", "long an", "bà rịa", "vũng tàu"]
        if any(keyword in address_lower for keyword in suburban_keywords):
            delivery_days = random.randint(1, 2)
            delivery_date = datetime.now() + timedelta(days=delivery_days)
            return delivery_date.strftime("%d/%m/%Y"), "trong vòng 1-2 ngày"

        delivery_days = random.randint(2, 5)
        delivery_date = datetime.now() + timedelta(days=delivery_days)
        return delivery_date.strftime("%d/%m/%Y"), "trong vòng 2-5 ngày làm việc"

    def _calculate_total_price(self, products):
        from src.services.price_utils import convert_usd_to_vnd

        total = 0
        for product in products:
            if "price" not in product:
                # Giá mặc định trong USD (giả sử nếu không có giá)
                price_usd = random.randint(40, 400)  # $40-$400
                price_vnd = convert_usd_to_vnd(price_usd)
                product["price"] = price_vnd
            else:
                # Nếu đã có giá (USD), chuyển sang VND
                price_vnd = convert_usd_to_vnd(product["price"])
                product["price"] = price_vnd

            total += product["price"] * product["quantity"]

        return total

    def format_price(self, price):
        # Import tiện ích xử lý giá
        from src.services.price_utils import format_price_usd_to_vnd
        return format_price_usd_to_vnd(price)

    def create_order(self, customer_info, products):
        order_id = self._generate_order_id()
        delivery_date, delivery_time = self._estimate_delivery_time(
            customer_info.get('address', ''))
        total_price = self._calculate_total_price(products)

        order = {
            "order_id": order_id,
            "customer": customer_info,
            "products": products,
            "total_price": total_price,
            "order_date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "delivery_date": delivery_date,
            "delivery_time": delivery_time,
            "status": "Chờ xác nhận",
            "payment_method": "",
            "payment_status": "Chưa thanh toán"
        }

        self.orders[order_id] = order

        return order

    async def process_order_with_info(self, customer_info, product_info):
        try:
            from src.services.price_utils import format_price_usd_to_vnd, convert_usd_to_vnd

            # Tạo đơn hàng trong hệ thống
            products = [{"name": product_info["name"],
                         "quantity": product_info.get("quantity", 1)}]
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
            confirmation = f"""
                Xác nhận đơn hàng #{order['order_id']}

                Thông tin khách hàng:
                - Tên: {customer_info['name']}
                - Số điện thoại: {customer_info['phone']}
                - Địa chỉ giao hàng: {customer_info['address']}

                Sản phẩm:
                {product_details}
                Tổng tiền: {self.format_price(total_price)}

                Thời gian giao hàng dự kiến: {order['delivery_date']} ({order['delivery_time']})

                Phương thức thanh toán:
                {payment_options}

                Cảm ơn bạn đã mua sắm tại TechPlus! Chúng tôi sẽ liên hệ với bạn để xác nhận đơn hàng trong thời gian sớm nhất.
                """
            return confirmation

        except Exception as e:
            print(f"Error processing order: {e}")
            return f"Xin lỗi, đã xảy ra lỗi khi xử lý đơn hàng. Vui lòng thử lại hoặc liên hệ với chúng tôi qua số hotline 1900-TECHPLUS."

    async def handle_query(self, query: str, language: str = "vi"):
        try:
            # Kiểm tra xem có phải đang muốn mua cấu hình vừa tư vấn không
            is_ordering_pc, confidence, reasoning = await self.detect_advised_pc_intent(query)

            if is_ordering_pc and confidence >= 0.7 and self.recently_advised_pc:
                product_info = ""
                for product in self.recently_advised_products:
                    name = product.get('name', '')
                    quantity = product.get('quantity', 1)
                    if name:
                        product_info += f"- {name} x{quantity}\n"

                response_with_signal = {
                    "content": f"Bạn muốn đặt hàng cấu hình PC vừa được tư vấn:\n{product_info}\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                    "show_order_form": True,
                    "products": self.recently_advised_products
                }
                return response_with_signal

            # Trích xuất sản phẩm từ văn bản đầu vào
            extracted_products = await self.extract_product_from_text(query)

            # Nếu không trích xuất được sản phẩm từ đầu vào
            if not extracted_products:
                # Kiểm tra nếu có các từ khóa liên quan đến đặt hàng
                order_keywords = ["đặt hàng", "mua ngay",
                                  "order", "thanh toán", "mua", "đặt", "lấy"]
                config_keywords = ["cấu hình", "pc",
                                   "máy tính", "bộ máy", "hệ thống"]

                has_order_intent = any(keyword in query.lower()
                                       for keyword in order_keywords)
                has_config_intent = any(keyword in query.lower()
                                        for keyword in config_keywords)

                if has_order_intent and has_config_intent and self.recently_advised_pc:
                    # Sử dụng cấu hình PC được tư vấn gần nhất
                    products = self.recently_advised_products
                    product_info = ""
                    for product in products:
                        name = product.get('name', '')
                        quantity = product.get('quantity', 1)
                        if name:
                            product_info += f"- {name} x{quantity}\n"

                    response_with_signal = {
                        "content": f"Bạn muốn đặt hàng sản phẩm đã được tư vấn:\n{product_info}\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                        "show_order_form": True,
                        "products": products
                    }
                    return response_with_signal
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

                response_with_signal = {
                    "content": f"Bạn đã chọn sản phẩm:\n{product_info}\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                    "show_order_form": True,
                    "products": extracted_products
                }

                return response_with_signal

        except Exception as e:
            print(f"Error in OrderProcessorAgent.handle_query: {e}")
            return f"Xin lỗi, tôi đang gặp sự cố khi xử lý đơn đặt hàng. Vui lòng thử lại sau hoặc liên hệ trực tiếp với cửa hàng qua hotline 1900-TECHPLUS."
