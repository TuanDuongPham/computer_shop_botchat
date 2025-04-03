from agents import Agent, Runner, FunctionTool, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from src.config import OPENAI_MODEL, OPENAI_API_KEY
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

        # Create agent using OpenAI Agent SDK
        self.agent = Agent(
            name="OrderProcessor",
            model=self.model_client,
            handoff_description="Specialist agent for order processing",
            handoffs=[self.format_price, self.create_order],
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
            price = random.randint(100000, 10000000)
            product["price"] = price
            total += price * product["quantity"]

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
            products = self._extract_product_info(query)

            if not products:
                return "Bạn muốn đặt sản phẩm gì? Vui lòng cung cấp thêm thông tin về sản phẩm bạn muốn mua."

            if products:
                product_info = ""
                for product in products:
                    quantity = product.get("quantity", 1)
                    product_info += f"- {product['name']} x{quantity}\n"

                response_with_signal = {
                    "content": f"Bạn đã chọn sản phẩm:\n{product_info}\nVui lòng cung cấp thông tin cá nhân để hoàn tất đơn hàng.",
                    "show_order_form": True,
                    "products": products
                }

                return response_with_signal
            else:
                return "Xin lỗi, tôi không thể xác định sản phẩm bạn muốn đặt. Vui lòng nêu rõ tên sản phẩm và số lượng bạn muốn mua."

        except Exception as e:
            print(f"Error in OrderProcessorAgent.handle_query: {e}")
            return f"Xin lỗi, tôi đang gặp sự cố khi xử lý đơn đặt hàng. Vui lòng thử lại sau hoặc liên hệ trực tiếp với cửa hàng qua hotline 1900-TECHPLUS."

    def create_order_from_form(self, form_data, products):
        customer_info = {
            "name": form_data.get("customer_name", ""),
            "phone": form_data.get("customer_phone", ""),
            "address": form_data.get("customer_address", "")
        }

        order = self.create_order(customer_info, products)

        product_details = ""
        total_price = 0

        for product in order["products"]:
            price = self.format_price(product["price"])
            product_details += f"- {product['name']} x{product['quantity']}: {price}\n"
            total_price += product["price"] * product["quantity"]

        order_confirmation = f"""
            Đơn hàng #{order['order_id']} đã được tạo thành công!

            Thông tin khách hàng:
            - Tên: {customer_info['name']}
            - SĐT: {customer_info['phone']}
            - Địa chỉ: {customer_info['address']}

            Sản phẩm:
            {product_details}
            Tổng tiền: {self.format_price(total_price)}

            Thời gian giao hàng dự kiến: {order['delivery_date']} ({order['delivery_time']})

            Phương thức thanh toán:
            - Thanh toán khi nhận hàng (COD)
            - Chuyển khoản ngân hàng
            - Thanh toán qua ví điện tử

            Cảm ơn bạn đã mua sắm tại TechPlus!
                """

        return {
            "order_id": order['order_id'],
            "confirmation": order_confirmation,
            "total_price": total_price
        }
