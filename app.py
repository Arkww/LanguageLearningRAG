import streamlit as st
from RAG import ask_rag  
from huggingface_hub import whoami
print(whoami())
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RAG Language Assistant",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- CUSTOM STYLES ---
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5em;
            color: #3F72AF;
            margin-bottom: 0.2em;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #808080;
            margin-bottom: 2em;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #A9A9A9;
            margin-top: 3em;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-title">🌍 RAG Language Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything about languages, translations, grammar, or vocabulary!</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/clouds/100/language.png", use_column_width=True)
st.sidebar.title("🧠 About This App")
st.sidebar.info(
    """
    This app uses **Retrieval-Augmented Generation (RAG)** to answer questions related to:
    
    - Language meanings
    - Translations
    - Grammar explanations
    - Vocabulary and usage
    """
)
st.sidebar.markdown("🔍 Powered by **LangChain**, **FAISS**, and **Transformers**.")

# --- MAIN FORM ---
with st.form("query_form"):
    query = st.text_input("💬 Ask your language-related question here:")
    submitted = st.form_submit_button("🚀 Ask")

if submitted:
    if query.strip():
        with st.spinner("Thinking... 🤔"):
            response = ask_rag(query.strip())
        st.success("✅ Answer:")
        st.write(response)
    else:
        st.warning("⚠️ Please enter a valid question.")

# --- FOOTER ---
st.markdown('<div class="footer">© 2025 - Mathieu Jay | Built with ❤️ using Streamlit</div>', unsafe_allow_html=True)
