import streamlit as st
from RAG import ask_rag  

# Page configuration
st.set_page_config(
    page_title="RAG Language Assistant",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Main title
st.title("🌍 RAG Language Assistant")


# Sidebar
st.sidebar.title("About")

st.sidebar.markdown("### Instructions")
st.sidebar.markdown(
    """
    - **Ask a question** about a language.
    - The system will provide an answer based on the knowledge it has.
    """
)

query = st.text_input("Ask a question about a language:")
if st.button("Send"):
    if query:
        response = ask_rag(query) 
        st.success("✅ Answer received:")
        st.write(response)
    else:
        st.warning("⚠️ Please enter a question before clicking Send.")

# Footer
st.markdown("---")
st.markdown("© 2025 - Mathieu Jay")