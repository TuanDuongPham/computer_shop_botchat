import streamlit as st
import requests
import uuid
import time

API_URL = "http://127.0.0.1:8000"
POLLING_INTERVAL = 2

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

if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

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


def poll_for_response():
    try:
        response = requests.get(
            f"{API_URL}/poll/{st.session_state.session_id}")

        if response.status_code == 200:
            data = response.json()

            if data.get("has_response", False):
                assistant_message = data.get("assistant_message", {})

                # Check if this is a new message we haven't seen before
                if assistant_message and assistant_message.get("content"):
                    current_messages = [m.get(
                        "content", "") for m in st.session_state.messages if m.get("role") == "assistant"]

                    if assistant_message.get("content") not in current_messages:
                        st.session_state.messages.append(assistant_message)
                        st.session_state.waiting_for_response = False
                        return True

        return False

    except Exception as e:
        st.error(f"Error polling for response: {str(e)}")
        st.session_state.waiting_for_response = False
        return False


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

if st.session_state.waiting_for_response:
    with st.spinner("Đang chờ phản hồi..."):
        has_new_response = poll_for_response()
        if has_new_response:
            st.experimental_rerun()

# Chat input
user_query = st.chat_input("Nhập câu hỏi của bạn...")

if user_query and not st.session_state.waiting_for_response:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.write(user_query)

    with st.chat_message("assistant", avatar="⏳"):
        message_placeholder = st.empty()
        message_placeholder.write("Đang xử lý...")

    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "query": user_query,
                "session_id": st.session_state.session_id
            }
        )

        if response.status_code == 200:
            st.session_state.waiting_for_response = True
            st.session_state.last_query_time = time.time()
            st.experimental_rerun()
        else:
            message_placeholder.error(
                f"Lỗi: {response.status_code} - {response.text}")

    except Exception as e:
        message_placeholder.error(f"Lỗi kết nối: {str(e)}")

# Sidebarn
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
    st.caption(f"Session ID: {st.session_state.session_id}")

    # Button to clear chat
    if st.button("Xóa cuộc trò chuyện"):
        st.session_state.messages = []
        st.session_state.waiting_for_response = False
        st.rerun()
