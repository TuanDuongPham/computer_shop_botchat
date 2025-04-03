from src.agents.agent_router import AgentRouter
from src.agents.product_advisor import ProductAdvisorAgent
from src.agents.policy_advisor import PolicyAdvisorAgent
from src.agents.general_advisor import GeneralAdvisorAgent
import streamlit as st
import uuid
import asyncio
import threading
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Khởi tạo session state TRƯỚC KHI nhập bất kỳ module tùy chỉnh nào
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# Cấu hình trang
st.set_page_config(
    page_title="TechPlus Hardware Advisor",
    page_icon="🔧",
    layout="wide"
)

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

# Define agent emoji icons
agent_icons = {
    "GeneralAdvisor": "🤖",
    "ProductAdvisor": "💻",
    "PolicyAdvisor": "📜",
    "PCBuilder": "🛠️",
    "OrderProcessor": "🛒",
    "CustomerService": "👤"
}

# Define agent names
agent_names = {
    "GeneralAdvisor": "Trợ lý chung",
    "ProductAdvisor": "Chuyên gia linh kiện",
    "PolicyAdvisor": "Tư vấn chính sách",
    "PCBuilder": "Xây dựng cấu hình",
    "OrderProcessor": "Đặt hàng",
    "CustomerService": "Dịch vụ khách hàng"
}

# Page header
st.title("🔧 TechPlus Hardware Advisor")
st.markdown("Chào mừng bạn đến với hệ thống tư vấn của TechPlus! Hãy hỏi tôi về linh kiện, chính sách của cửa hàng, hoặc để tôi giúp bạn xây dựng cấu hình PC phù hợp.")

# Hàm khởi tạo các agent - chỉ chạy 1 lần khi ứng dụng khởi động


def initialize_agents():
    if st.session_state.initialized:
        return

    with st.spinner("Đang khởi tạo hệ thống trợ lý..."):
        # Khởi tạo router
        st.session_state.agent_router = AgentRouter()

        # Khởi tạo các agent cụ thể
        st.session_state.agents = {
            "product_advisor": ProductAdvisorAgent(),
            "policy_advisor": PolicyAdvisorAgent(),
            "general": GeneralAdvisorAgent(),
        }

        # Set agent mặc định
        st.session_state.default_agent = st.session_state.agents["general"]

        # Đánh dấu đã khởi tạo
        st.session_state.initialized = True

# Hàm xử lý câu hỏi từ người dùng


async def process_query(query, language="vi"):
    try:
        # Xác định loại agent cần dùng
        agent_type = await st.session_state.agent_router.route_query(query)

        # Lấy agent phù hợp
        agent_instance = st.session_state.agents.get(agent_type)
        if not agent_instance:
            # Fallback sang agent mặc định
            agent_instance = st.session_state.default_agent

        # Xử lý câu hỏi với agent được chọn
        response = await agent_instance.handle_query(query, language)

        # Tạo response object với metadata của agent
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

# Wrapper để chạy async function trong Streamlit


def run_async_query(query):
    try:
        st.session_state.processing = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(process_query(query))
        loop.close()

        # Đảm bảo messages đã được khởi tạo
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Thêm response vào messages
        st.session_state.messages.append(response)
    except Exception as e:
        st.error(f"Lỗi xử lý câu hỏi: {str(e)}")
    finally:
        st.session_state.processing = False


# Khởi tạo agents khi ứng dụng khởi động
initialize_agents()

# Display chat messages
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

                    # Show which agent responded
                    agent_name = agent_names.get(sender, sender)
                    st.caption(f"Trả lời bởi: {agent_name}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])

# Chat input
if st.session_state.processing:
    # Disable chat input while processing
    with st.chat_message("assistant", avatar="⏳"):
        st.write("Đang xử lý câu hỏi của bạn...")
else:
    # Enable chat input when not processing
    user_query = st.chat_input("Nhập câu hỏi của bạn...")

    if user_query:
        # Add user message to chat
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.session_state.messages.append(
            {"role": "user", "content": user_query})
        st.session_state.processing = True
        st.rerun()  # Rerun để hiển thị spinner

# Nếu đang xử lý và chưa có kết quả, thực hiện xử lý
if st.session_state.processing:
    with st.spinner("Đang xử lý..."):
        # Xử lý câu hỏi trực tiếp, không qua thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Lấy câu hỏi cuối cùng từ user
        user_messages = [
            m for m in st.session_state.messages if m["role"] == "user"]
        if user_messages:
            latest_query = user_messages[-1]["content"]
            response = loop.run_until_complete(process_query(latest_query))
            loop.close()

            st.session_state.messages.append(response)

        st.session_state.processing = False
        st.rerun()  # Rerun để hiển thị kết quả

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

    # Debug information
    with st.expander("Thông tin Debug"):
        st.caption(f"Session ID: {st.session_state.session_id}")
        st.caption(
            f"Trạng thái: {'Đang xử lý' if st.session_state.processing else 'Sẵn sàng'}")
        st.caption(f"Agents đã khởi tạo: {st.session_state.initialized}")

        if st.button("Khởi động lại agents"):
            st.session_state.initialized = False
            initialize_agents()
            st.success("Đã khởi động lại các agents thành công!")

    # Button to clear chat
    if st.button("Xóa cuộc trò chuyện"):
        st.session_state.messages = []
        st.rerun()
