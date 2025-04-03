import streamlit as st
import requests
import uuid
import time

API_URL = "http://127.0.0.1:8000"

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

if "last_poll_time" not in st.session_state:
    st.session_state.last_poll_time = 0

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
    """Kiểm tra phản hồi mới từ server và trả về True nếu có phản hồi mới"""
    try:
        current_time = time.time()
        # Chỉ poll mỗi 1 giây để tránh quá tải
        if current_time - st.session_state.last_poll_time < 1:
            return False

        st.session_state.last_poll_time = current_time

        response = requests.get(
            f"{API_URL}/poll/{st.session_state.session_id}",
            timeout=2
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("has_response", False):
                assistant_message = data.get("assistant_message", {})
                if assistant_message and assistant_message.get("content"):
                    # Kiểm tra xem message này đã có trong danh sách chưa
                    current_contents = [msg.get(
                        "content", "") for msg in st.session_state.messages if msg.get("role") == "assistant"]
                    if assistant_message.get("content") not in current_contents:
                        st.session_state.messages.append(assistant_message)
                        st.session_state.waiting_for_response = False
                        return True
        return False
    except Exception as e:
        print(f"Poll error: {str(e)}")
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

# Chat input
user_query = st.chat_input("Nhập câu hỏi của bạn...")

# Auto refresh để liên tục poll khi đang chờ phản hồi
if st.session_state.waiting_for_response:
    # Hiển thị indicator
    with st.chat_message("assistant", avatar="⏳"):
        st.write("Đang xử lý...")

    # Kiểm tra phản hồi mới
    has_response = poll_for_response()
    if has_response:
        st.rerun()

    # Tạo thời gian mới
    st.markdown(
        f"""
        <p id="counter" style="display:none;">{time.time()}</p>
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, 2000);
        </script>
        """,
        unsafe_allow_html=True
    )

if user_query and not st.session_state.waiting_for_response:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.write(user_query)

    # Show processing indicator
    with st.chat_message("assistant", avatar="⏳"):
        st.write("Đang xử lý...")

    try:
        # Send request to server
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "query": user_query,
                "session_id": st.session_state.session_id
            },
            timeout=5
        )

        if response.status_code == 200:
            st.session_state.waiting_for_response = True
            st.rerun()
        else:
            error_msg = f"Lỗi: {response.status_code} - {response.text}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
            st.rerun()

    except Exception as e:
        error_msg = f"Lỗi kết nối: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg
        })
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

    # Debug information
    with st.expander("Thông tin"):
        st.caption(f"Session ID: {st.session_state.session_id}")
        st.caption(
            f"Trạng thái: {'Đang chờ phản hồi' if st.session_state.waiting_for_response else 'Sẵn sàng'}")
        st.caption(
            f"Thời gian poll gần nhất: {st.session_state.last_poll_time}")

        if st.button("Đặt lại trạng thái"):
            st.session_state.waiting_for_response = False
            st.rerun()

    # Button to clear chat
    if st.button("Xóa cuộc trò chuyện"):
        st.session_state.messages = []
        st.session_state.waiting_for_response = False
        st.rerun()
