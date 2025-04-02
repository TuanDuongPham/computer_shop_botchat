import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.title("🔧 Hardware Advisor Chatbot")
st.write("Ask me anything about GPUs, RAM, PSUs, Storage, Coolers, and Motherboards!")

# Chat input
user_query = st.text_input("Your question:")

if st.button("Ask"):
    if user_query:
        response = requests.post(API_URL, json={"query": user_query})
        bot_reply = response.json().get("response", "Error retrieving response.")
        st.write(f"**Chatbot:** {bot_reply}")
    else:
        st.warning("Please enter a question!")
