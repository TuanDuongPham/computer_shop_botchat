from agents import Agent, Runner, FunctionTool, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from src.config import OPENAI_MODEL, OPENAI_API_KEY
from src.services.shared_state import SharedStateService
from src.services.price_utils import format_price_usd_to_vnd
import re
import json
import time
from datetime import datetime, timedelta
import random


class OrderProcessorAgent:
    def __init__(self):
        self.model_client = OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY)
        )

        self.shared_state = SharedStateService()

        self.agent = Agent(
            name="OrderProcessor",
            model=self.model_client,
            handoff_description="Specialist agent for order processing",
            handoffs=[self.format_price, self.create_order,
                      self.extract_product_from_text, self.detect_advised_pc_intent],
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

    async def detect_advised_pc_intent(self, query):
        try:
            recently_advised_products = self.shared_state.get_recently_advised_products()
            is_advised_pc = self.shared_state.is_recently_advised_pc()

            if not recently_advised_products:
                return False, 0.0, "Không có sản phẩm tư vấn gần đây"

            product_list = ""
            for product in recently_advised_products:
                product_list += f"- {product.get('name', 'Unknown')} ({product.get('category', 'Unknown')})\n"

            prompt = f"""
            Phân tích đoạn văn bản sau để xác định xem người dùng có ý định đặt mua sản phẩm đã được tư vấn trước đó không:
            "{query}"
            
            Các sản phẩm đã được tư vấn gần đây:
            {product_list}
            
            Đây {'' if is_advised_pc else 'không'} là một cấu hình PC đầy đủ.
            
            Cần phân biệt rõ hai trường hợp:
            1. Người dùng muốn đặt MỘT sản phẩm cụ thể (ví dụ: "đặt hàng chip Intel i5 14600X")
            2. Người dùng muốn đặt TẤT CẢ sản phẩm đã tư vấn (ví dụ: "đặt hàng cấu hình này")
            
            Các từ khóa liên quan đến đặt hàng: mua, đặt, order, thanh toán, lấy, chốt đơn, xác nhận, đồng ý, ok
            Các từ khóa liên quan đến xác nhận toàn bộ cấu hình: cấu hình, pc, máy tính, bộ máy, tất cả, toàn bộ, những sản phẩm này
            Các từ khóa chỉ định một sản phẩm: sản phẩm này, chip, card, ram, cpu, ổ cứng, kèm theo tên cụ thể
            
            Phân tích ngữ cảnh, đánh giá và trả về kết quả theo định dạng JSON:
            {{
                "is_ordering": true/false,
                "confidence": <điểm tin cậy từ 0.0 đến 1.0>,
                "reasoning": "<giải thích lý do>",
                "single_product": true/false,
                "mentioned_product": "<tên sản phẩm cụ thể nếu được nhắc đến>"
            }}
            """

            response = await Runner.run(
                Agent(
                    name="OrderIntentDetector",
                    model=self.model_client,
                    instructions="Xác định ý định đặt hàng từ văn bản đầu vào"
                ),
                [{"role": "user", "content": prompt}]
            )

            try:
                result_text = response.final_output

                import re
                json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}))*\}'
                json_match = re.search(json_pattern, result_text)

                if json_match:
                    result_json = json.loads(json_match.group(0))
                    is_ordering = result_json.get("is_ordering", False)
                    confidence = result_json.get("confidence", 0.0)
                    reasoning = result_json.get("reasoning", "")
                    single_product = result_json.get("single_product", False)
                    mentioned_product = result_json.get(
                        "mentioned_product", "")

                    return is_ordering, confidence, reasoning, single_product, mentioned_product
                else:
                    order_keywords = ["mua", "đặt", "order",
                                      "thanh toán", "lấy", "chốt", "đồng ý", "ok"]
                    config_keywords = [
                        "cấu hình", "pc", "máy tính", "bộ máy", "như trên", "vừa rồi"]
                    specific_product_keywords = [
                        "cpu", "chip", "card", "ram", "i5", "i7", "i9", "ryzen"]

                    has_order_intent = any(k in query.lower()
                                           for k in order_keywords)
                    has_config_intent = any(k in query.lower()
                                            for k in config_keywords)
                    might_be_specific_product = any(
                        k in query.lower() for k in specific_product_keywords)

                    if has_order_intent and might_be_specific_product and not has_config_intent:
                        mentioned = ""
                        for keyword in specific_product_keywords:
                            if keyword in query.lower():
                                pattern = rf'{keyword}\s+([^\s.,;]+(?:\s+[^\s.,;]+)*)'
                                match = re.search(pattern, query.lower())
                                if match:
                                    mentioned = match.group(0)
                                    break

                        return True, 0.7, "Phát hiện ý định đặt một sản phẩm cụ thể", True, mentioned

                    if has_order_intent and (has_config_intent or is_advised_pc):
                        return True, 0.8, "Phát hiện ý định đặt hàng toàn bộ cấu hình", False, ""

                    return False, 0.0, "Không phát hiện ý định đặt hàng", False, ""
            except Exception as e:
                print(f"Lỗi khi phân tích JSON từ phản hồi: {e}")
                return False, 0.0, f"Lỗi phân tích: {e}"

        except Exception as e:
            print(f"Lỗi khi phát hiện ý định đặt PC: {e}")
            return False, 0.0, f"Lỗi tổng thể: {e}"

    async def extract_product_from_text(self, text):
        try:
            recently_advised_products = self.shared_state.get_recently_advised_products()

            recent_products_text = ""
            if recently_advised_products:
                recent_products_text = "Sản phẩm đã tư vấn gần đây:\n"
                for product in recently_advised_products:
                    recent_products_text += f"- {product.get('name')} ({product.get('category')})\n"

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
            - QUAN TRỌNG: Nếu văn bản chỉ đề cập đến một sản phẩm cụ thể (ví dụ: "Đặt hàng chip Intel i5 14600X"), hãy CHỈ trích xuất sản phẩm đó, không bao gồm các sản phẩm khác đã tư vấn trước đó.
            - Nếu văn bản đề cập đến một sản phẩm cụ thể từ danh sách đã tư vấn, CHỈ trả về sản phẩm đó, không bao gồm các sản phẩm khác.
            - Nếu văn bản chỉ thể hiện ý định mua hàng chung chung (như "Tôi muốn đặt hàng", "Đặt hàng ngay", "Mua sản phẩm") mà không chỉ định sản phẩm cụ thể và có sản phẩm đã tư vấn gần đây, hãy sử dụng thông tin từ tất cả sản phẩm đã tư vấn.
            - Nếu văn bản có đề cập đến loại sản phẩm (như "CPU", "RAM", "card đồ họa") nhưng không nêu cụ thể tên, và có sản phẩm tương ứng đã tư vấn gần đây, hãy chỉ sử dụng thông tin từ sản phẩm đã tư vấn thuộc loại đó.
            - Nếu không có sản phẩm cụ thể nào được đề cập, hãy trả về danh sách trống [].
            - Nếu văn bản đề cập đến "cấu hình", "bộ máy", "PC đầy đủ", hoặc sử dụng đại từ "tất cả", "toàn bộ" khi nhắc đến sản phẩm đã tư vấn, hãy trả về toàn bộ sản phẩm đã tư vấn.
            
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
                raw_text = response.final_output
                start_idx = raw_text.find('[')
                end_idx = raw_text.rfind(']') + 1
                if start_idx == -1 or end_idx == 0:
                    if recently_advised_products and any(keyword in text.lower() for keyword in ["mua", "đặt", "order", "lấy", "chốt"]):
                        return recently_advised_products
                    return []

                json_text = raw_text[start_idx:end_idx]
                products = json.loads(json_text)

                if not products and recently_advised_products:
                    if any(keyword in text.lower() for keyword in ["mua", "đặt", "order", "thanh toán", "lấy", "chốt"]):
                        # Kiểm tra nếu có nhắc đến "cấu hình" hoặc "PC"
                        pc_keywords = ["cấu hình", "pc", "máy tính",
                                       "bộ máy", "như trên", "vừa rồi", "sản phẩm"]
                        if any(keyword in text.lower() for keyword in pc_keywords):
                            return recently_advised_products

                if products:
                    for product in products:
                        category_keywords = {
                            "CPU": ["cpu", "chip", "bộ xử lý", "vi xử lý"],
                            "GPU": ["gpu", "card đồ họa", "vga"],
                            "RAM": ["ram", "bộ nhớ"],
                            "Motherboard": ["mainboard", "bo mạch chủ"],
                            "Storage": ["ổ cứng", "ssd", "hdd"],
                            "PSU": ["nguồn", "psu", "power supply"],
                            "Case": ["case", "vỏ máy tính"],
                            "Cooling": ["tản nhiệt", "fan", "quạt"]
                        }

                        product_name = product.get("name", "").lower()

                        for category, keywords in category_keywords.items():
                            if any(keyword in product_name for keyword in keywords) and len(product_name.split()) <= 2:
                                matching_products = [
                                    p for p in recently_advised_products if p.get("category") == category]
                                if matching_products:
                                    product["name"] = matching_products[0]["name"]
                                    break

                enhanced_products = []
                for product in products:
                    product_name = product.get("name")
                    quantity = product.get("quantity", 1)

                    matching_products = [
                        p for p in recently_advised_products if p.get("name") == product_name]
                    if matching_products:
                        full_product = matching_products[0].copy()
                        full_product["quantity"] = quantity
                        enhanced_products.append(full_product)
                    else:
                        for advised_product in recently_advised_products:
                            advised_name = advised_product.get(
                                "name", "").lower()
                            if product_name.lower() in advised_name or advised_name in product_name.lower():
                                full_product = advised_product.copy()
                                full_product["quantity"] = quantity
                                enhanced_products.append(full_product)
                                break
                        else:
                            enhanced_products.append({
                                "name": product_name,
                                "quantity": quantity,
                                "category": "Unknown",
                                "price": 0
                            })

                return enhanced_products if enhanced_products else products
            except Exception as e:
                print(f"Lỗi khi phân tích JSON: {e}")
                if recently_advised_products and any(keyword in text.lower() for keyword in ["mua", "đặt", "order", "lấy"]):
                    return recently_advised_products
                return []
        except Exception as e:
            print(f"Lỗi khi trích xuất sản phẩm: {e}")
            recently_advised_products = self.shared_state.get_recently_advised_products()
            if recently_advised_products and any(keyword in text.lower() for keyword in ["mua", "đặt", "order", "lấy"]):
                return recently_advised_products
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

    def _extract_product_info(self, query):
        products = []

        product_patterns = [
            r'(?:mua|đặt|order)\s+([^\.]+)',
            r'sản\s+phẩm(?:\s+là)?\s+([^\.]+)'
        ]

        for pattern in product_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                product_text = match.group(1).strip()
                if len(product_text) > 5:
                    products.append({"name": product_text, "quantity": 1})

        quantity_patterns = [
            r'(\d+)\s+(?:cái|chiếc|sản phẩm)',
            r'số\s+lượng(?:\s+là)?\s+(\d+)'
        ]

        for pattern in quantity_patterns:
            match = re.search(pattern, query)
            if match and products:
                products[-1]["quantity"] = int(match.group(1))

        return products

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
        total = 0
        for product in products:
            if "price" in product and product["price"]:
                price = float(product["price"])
            else:
                price = random.randint(100000, 10000000)

            product["price"] = price
            quantity = product.get("quantity", 1)
            total += price * quantity

        return total

    def format_price(self, price):
        return "{:,.0f}".format(price).replace(",", ".") + "đ"

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
            products = [{"name": product_info["name"],
                         "quantity": product_info.get("quantity", 1)}]
            order = self.create_order(customer_info, products)

            payment_options = "\n".join(
                [f"- {method}" for method in self.payment_methods])

            product_details = ""
            total_price = 0

            for product in order["products"]:
                price = self.format_price(product["price"])
                product_details += f"- {product['name']} x{product['quantity']}: {price}\n"
                total_price += product["price"] * product["quantity"]

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

    def create_order_from_form(self, customer_info, products):
        try:
            order = self.create_order(customer_info, products)

            payment_options = "\n".join(
                [f"- {method}" for method in self.payment_methods])

            product_details = ""
            total_price = 0

            for product in products:
                quantity = product.get("quantity", 1)
                price = product.get("price", 0)
                formatted_price = self.format_price(price)
                product_details += f"- {product['name']} x{quantity}: {formatted_price}\n"
                total_price += price * quantity

            formatted_total = self.format_price(total_price)

            confirmation = f"""
            Xác nhận đơn hàng #{order['order_id']}

            Thông tin khách hàng:
            - Tên: {customer_info['customer_name']}
            - Số điện thoại: {customer_info['customer_phone']}
            - Địa chỉ giao hàng: {customer_info['customer_address']}

            Sản phẩm:
            {product_details}
            Tổng tiền: {formatted_total}

            Thời gian giao hàng dự kiến: {order['delivery_date']} ({order['delivery_time']})

            Phương thức thanh toán:
            {payment_options}

            Cảm ơn bạn đã mua sắm tại TechPlus! Chúng tôi sẽ liên hệ với bạn để xác nhận đơn hàng trong thời gian sớm nhất.
            """

            return {
                "order": order,
                "confirmation": confirmation
            }
        except Exception as e:
            print(f"Error in create_order_from_form: {e}")
            return {
                "order": None,
                "confirmation": f"Xin lỗi, đã xảy ra lỗi khi xử lý đơn hàng. Vui lòng thử lại hoặc liên hệ với chúng tôi qua số hotline 1900-TECHPLUS. Lỗi: {str(e)}"
            }

    async def handle_query(self, query: str, language: str = "vi"):
        try:
            intent_result = await self.detect_advised_pc_intent(query)

            if isinstance(intent_result, tuple):
                if len(intent_result) >= 5:
                    is_ordering_pc, confidence, reasoning, single_product, mentioned_product = intent_result
                elif len(intent_result) >= 3:
                    is_ordering_pc, confidence, reasoning = intent_result
                    single_product = False
                    mentioned_product = ""
                else:
                    is_ordering_pc, confidence, reasoning = False, 0.0, "Định dạng kết quả không hợp lệ"
                    single_product = False
                    mentioned_product = ""
            else:
                is_ordering_pc, confidence, reasoning = False, 0.0, "Định dạng kết quả không hợp lệ"
                single_product = False
                mentioned_product = ""

            if is_ordering_pc and confidence >= 0.7:
                print(
                    f"Phát hiện ý định đặt hàng sản phẩm đã tư vấn: {reasoning}")
                recently_advised_products = self.shared_state.get_recently_advised_products()

                if recently_advised_products:
                    if single_product and mentioned_product:
                        print(
                            f"Đang tìm kiếm sản phẩm cụ thể: {mentioned_product}")
                        matching_products = []
                        for product in recently_advised_products:
                            product_name = product.get('name', '').lower()
                            if mentioned_product.lower() in product_name or any(part.lower() in product_name for part in mentioned_product.lower().split()):
                                matching_products.append(product)

                        if matching_products:
                            product_info = ""
                            for product in matching_products:
                                quantity = product.get("quantity", 1)
                                price = format_price_usd_to_vnd(
                                    product.get("price", 0))
                                product_info += f"- {product['name']} x{quantity} ({price})\n"

                            response_with_signal = {
                                "content": f"Bạn muốn đặt hàng sản phẩm:\n{product_info}\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                                "show_order_form": True,
                                "products": matching_products
                            }
                            return response_with_signal

                    product_info = ""
                    for product in recently_advised_products:
                        quantity = product.get("quantity", 1)
                        price = format_price_usd_to_vnd(
                            product.get("price", 0))
                        product_info += f"- {product['name']} x{quantity} ({price})\n"

                    response_with_signal = {
                        "content": f"Bạn muốn đặt hàng với sản phẩm đã được tư vấn:\n{product_info}\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                        "show_order_form": True,
                        "products": recently_advised_products
                    }
                    return response_with_signal

                elif has_order_intent:
                    return "Bạn muốn đặt sản phẩm gì? Vui lòng cung cấp thêm thông tin về sản phẩm bạn muốn mua."
                else:
                    return "Tôi có thể hỗ trợ bạn đặt hàng các sản phẩm linh kiện máy tính. Vui lòng cho biết bạn muốn mua sản phẩm gì?"

            extracted_products = await self.extract_product_from_text(query)

            if not extracted_products:
                order_keywords = ["đặt hàng", "mua ngay", "order",
                                  "thanh toán", "mua", "lấy", "chốt đơn"]
                pc_keywords = ["cấu hình", "pc", "máy tính", "bộ máy",
                               "như trên", "vừa rồi", "như vậy"]

                has_order_intent = any(keyword in query.lower()
                                       for keyword in order_keywords)
                has_pc_reference = any(keyword in query.lower()
                                       for keyword in pc_keywords)

                recently_advised_products = self.shared_state.get_recently_advised_products()

                if has_order_intent and (has_pc_reference or self.shared_state.is_recently_advised_pc()) and recently_advised_products:
                    product_info = ""
                    for product in recently_advised_products:
                        quantity = product.get("quantity", 1)
                        price = format_price_usd_to_vnd(
                            product.get("price", 0))
                        product_info += f"- {product['name']} x{quantity} ({price})\n"

                    response_with_signal = {
                        "content": f"Bạn muốn đặt hàng với sản phẩm đã được tư vấn trước đó:\n{product_info}\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                        "show_order_form": True,
                        "products": recently_advised_products
                    }
                    return response_with_signal
        except Exception as e:
            print(f"Error in OrderProcessorAgent.handle_query: {e}")
            return f"Xin lỗi, tôi đang gặp sự cố khi xử lý đơn đặt hàng. Vui lòng thử lại sau hoặc liên hệ trực tiếp với cửa hàng qua hotline 1900-TECHPLUS."
