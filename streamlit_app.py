import streamlit as st
from chatbot import NLPChatbot

st.set_page_config(page_title="NLP Chatbot", page_icon="🤖")

st.title("🤖 NLP Chatbot")
st.caption("NLTK + TF-IDF + Logistic Regression")

@st.cache_resource
def load_bot():
    return NLPChatbot()

bot = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_text = st.chat_input("Type your message...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    try:
        response = bot.get_response(user_text)
    except AttributeError:
        response = bot.respond(user_text)

    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("About")
    st.write("This chatbot uses Natural Language Processing with NLTK, TF-IDF and Logistic Regression.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
