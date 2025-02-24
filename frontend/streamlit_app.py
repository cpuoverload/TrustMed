import streamlit as st
import requests
import json
from typing import List

st.set_page_config(page_title="RAG Chat Assistant", page_icon="🤖", layout="wide")

API_URL = "http://localhost:8000"

# Initialize conversation state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS styles
st.markdown(
    """
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #e6f3ff;
}
.chat-message.assistant {
    background-color: #f7f7f7;
}
.chat-message .message-content {
    display: flex;
    margin-top: 0.5rem;
}
.chat-message .avatar {
    width: 40px;
    height: 40px;
    margin-right: 1rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}
.chat-message .content {
    flex-grow: 1;
}
</style>
""",
    unsafe_allow_html=True,
)

# Display title
st.title("🤖 RAG Chat Assistant")
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.container():
        st.markdown(
            f"""
        <div class="chat-message {message['role']}">
            <div class="message-content">
                <div class="avatar">{' 👤' if message['role'] == 'user' else '🤖'}</div>
                <div class="content">{message['content']}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Input box
if prompt := st.chat_input("Enter your question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.container():
        st.markdown(
            f"""
        <div class="chat-message user">
            <div class="message-content">
                <div class="avatar">👤</div>
                <div class="content">{prompt}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Create an empty container for streaming the response
    with st.container():
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Send request to API
            response = requests.post(
                f"{API_URL}/chat",
                json={"question": prompt, "stream": True},
                stream=True,
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

            # Process raw response directly
            for line in response.iter_lines():
                if line:
                    # Decode and remove "data: " prefix
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]  # Skip "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            json_data = json.loads(data)
                            chunk = json_data.get("delta", "")
                            full_response += chunk

                            # Update displayed message
                            message_placeholder.markdown(
                                f"""
                            <div class="chat-message assistant">
                                <div class="message-content">
                                    <div class="avatar">🤖</div>
                                    <div class="content">{full_response}</div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        except json.JSONDecodeError:
                            continue

            # Add assistant message to history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

# Add clear conversation button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Display API status
try:
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        st.sidebar.success("API Status: Online")
    else:
        st.sidebar.error("API Status: Offline")
except:
    st.sidebar.error("API Status: Cannot Connect")
