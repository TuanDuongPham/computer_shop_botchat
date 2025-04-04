from src.agents.agent_router import AgentRouter
from src.agents.product_advisor import ProductAdvisorAgent
from src.agents.policy_advisor import PolicyAdvisorAgent
from src.agents.pc_builder import PCBuilderAgent
from src.agents.order_processor import OrderProcessorAgent
from src.agents.general_advisor import GeneralAdvisorAgent
import streamlit as st
import uuid
import asyncio
import threading
import os
import sys
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "agent_router" not in st.session_state:
    st.session_state.agent_router = None

if "agents" not in st.session_state:
    st.session_state.agents = {}

if "default_agent" not in st.session_state:
    st.session_state.default_agent = None

if "order_form_data" not in st.session_state:
    st.session_state.order_form_data = None

if "pending_order_products" not in st.session_state:
    st.session_state.pending_order_products = None

if "order_confirmation" not in st.session_state:
    st.session_state.order_confirmation = None

st.set_page_config(
    page_title="TechPlus Hardware Advisor",
    page_icon="🔧",
    layout="wide"
)


agent_icons = {
    "GeneralAdvisor": "🤖",
    "ProductAdvisor": "💻",
    "PolicyAdvisor": "📜",
    "PCBuilder": "🛠️",
    "OrderProcessor": "🛒",
    "CustomerService": "👤"
}

agent_names = {
    "GeneralAdvisor": "Trợ lý chung",
    "ProductAdvisor": "Chuyên gia linh kiện",
    "PolicyAdvisor": "Tư vấn chính sách",
    "PCBuilder": "Xây dựng cấu hình",
    "OrderProcessor": "Đặt hàng",
    "CustomerService": "Dịch vụ khách hàng"
}

st.title("🔧 TechPlus Hardware Advisor")
st.markdown("Chào mừng bạn đến với hệ thống tư vấn của TechPlus! Hãy hỏi tôi về linh kiện, chính sách của cửa hàng, hoặc để tôi giúp bạn xây dựng cấu hình PC phù hợp.")


def initialize_agents():
    if st.session_state.initialized:
        return

    with st.spinner("Đang khởi tạo hệ thống trợ lý..."):
        st.session_state.agent_router = AgentRouter()

        st.session_state.agents = {
            "product_advisor": ProductAdvisorAgent(),
            "policy_advisor": PolicyAdvisorAgent(),
            "pc_builder": PCBuilderAgent(),
            "order_processor": OrderProcessorAgent(),
            "general": GeneralAdvisorAgent(),
        }

        st.session_state.default_agent = st.session_state.agents["general"]

        st.session_state.initialized = True


async def process_query(query, language="vi"):
    try:
        agent_type = await st.session_state.agent_router.route_query(query)

        agent_instance = st.session_state.agents.get(agent_type)
        if not agent_instance:
            agent_instance = st.session_state.default_agent

        response = await agent_instance.handle_query(query, language)

        if agent_type in ["product_advisor", "pc_builder"] and hasattr(agent_instance, "recently_advised_products"):
            st.session_state.agent_router.set_recently_advised_products(
                agent_instance.recently_advised_products)

        if isinstance(response, dict) and "show_order_form" in response:
            st.session_state.pending_order_products = response.get(
                "products", [])

            agent_response = {
                "content": response["content"],
                "sender": agent_instance.agent.name
            }

            return {
                "role": "assistant",
                "content": response["content"],
                "agent_responses": [agent_response],
                "show_order_form": True
            }
        else:
            agent_response = {
                "content": response,
                "sender": agent_instance.agent.name
            }

            return {
                "role": "assistant",
                "content": response,
                "agent_responses": [agent_response]
            }

    except Exception as e:
        return {
            "role": "assistant",
            "content": f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn: {str(e)}",
            "agent_responses": [{
                "content": f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn: {str(e)}",
                "sender": "GeneralAdvisor"
            }]
        }


def run_async_query(query):
    try:
        st.session_state.processing = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(process_query(query))
        loop.close()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.session_state.messages.append(response)
    except Exception as e:
        st.error(f"Lỗi xử lý câu hỏi: {str(e)}")
    finally:
        st.session_state.processing = False


def render_order_form():
    last_message = st.session_state.messages[-1] if st.session_state.messages else None

    if last_message and last_message.get("show_order_form", False):
        with st.expander("📝 Thông tin đặt hàng", expanded=True):
            with st.form("order_form"):
                st.write("Vui lòng điền thông tin để hoàn tất đơn hàng:")
                customer_name = st.text_input("Họ và tên")
                customer_phone = st.text_input("Số điện thoại")
                customer_address = st.text_area("Địa chỉ giao hàng")

                submit_button = st.form_submit_button("Xác nhận đặt hàng")

                if submit_button:
                    if not customer_name or not customer_phone or not customer_address:
                        st.error("Vui lòng điền đầy đủ thông tin")
                    else:
                        st.session_state.order_form_data = {
                            "customer_name": customer_name,
                            "customer_phone": customer_phone,
                            "customer_address": customer_address
                        }

                        if st.session_state.pending_order_products:
                            order_processor = st.session_state.agents.get(
                                "order_processor")
                            if order_processor:
                                order_result = order_processor.create_order_from_form(
                                    st.session_state.order_form_data,
                                    st.session_state.pending_order_products
                                )

                                st.session_state.order_confirmation = order_result["confirmation"]

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": order_result["confirmation"],
                                    "agent_responses": [{
                                        "content": order_result["confirmation"],
                                        "sender": "OrderProcessor"
                                    }]
                                })

                                st.session_state.pending_order_products = None
                                st.rerun()


initialize_agents()

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message["content"])
    else:
        if "agent_responses" in message:
            for agent_response in message["agent_responses"]:
                sender = agent_response.get("sender", "GeneralAdvisor")
                content = agent_response.get("content", "")

                icon = agent_icons.get(sender, "🤖")
                with st.chat_message("assistant", avatar=icon):
                    st.write(content)

                    agent_name = agent_names.get(sender, sender)
                    st.caption(f"Trả lời bởi: {agent_name}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])

render_order_form()

# Chat input
if st.session_state.processing:
    # Disable chat input while processing
    with st.chat_message("assistant", avatar="⏳"):
        st.write("Đang xử lý câu hỏi của bạn...")
else:
    # Enable chat input when not processing
    user_query = st.chat_input("Nhập câu hỏi của bạn...")

    if user_query:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.session_state.messages.append(
            {"role": "user", "content": user_query})
        st.session_state.processing = True
        st.rerun()

if st.session_state.processing:
    with st.spinner("Đang xử lý..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        user_messages = [
            m for m in st.session_state.messages if m["role"] == "user"]
        if user_messages:
            latest_query = user_messages[-1]["content"]
            response = loop.run_until_complete(process_query(latest_query))
            loop.close()

            st.session_state.messages.append(response)

        st.session_state.processing = False
        st.rerun()

# Sidebar
with st.sidebar:
    st.header("Các chuyên gia TechPlus")

    st.markdown(f"""
    #### {agent_icons['GeneralAdvisor']} {agent_names['GeneralAdvisor']}
    Trả lời các câu hỏi chung về cửa hàng
    
    #### {agent_icons['ProductAdvisor']} {agent_names['ProductAdvisor']}
    Tư vấn và so sánh linh kiện máy tính
    
    #### {agent_icons['PolicyAdvisor']} {agent_names['PolicyAdvisor']}
    Thông tin về chính sách bảo hành, đổi trả
    
    #### {agent_icons['PCBuilder']} {agent_names['PCBuilder']}
    Xây dựng cấu hình PC theo nhu cầu
    
    #### {agent_icons['OrderProcessor']} {agent_names['OrderProcessor']}
    Hỗ trợ đặt hàng và xác nhận đơn hàng
    """)

    st.divider()

    st.markdown("**Thông tin cửa hàng**")
    st.markdown("""
    🏪 **TechPlus**  
    📍 123 Đường Công Nghệ, Q. Trung Tâm, TP.HCM  
    📞 1900-TECHPLUS  
    🌐 www.techplus.vn  
    ⏰ 08:00 - 21:00 (Thứ 2 - Chủ Nhật)""")

    st.divider()

    with st.expander("Thông tin Debug"):
        st.caption(f"Session ID: {st.session_state.session_id}")
        st.caption(
            f"Trạng thái: {'Đang xử lý' if st.session_state.processing else 'Sẵn sàng'}")
        st.caption(f"Agents đã khởi tạo: {st.session_state.initialized}")

        if st.button("Khởi động lại agents"):
            st.session_state.initialized = False
            initialize_agents()
            st.success("Đã khởi động lại các agents thành công!")

    if st.button("Xóa cuộc trò chuyện"):
        st.session_state.messages = []
        st.rerun()
