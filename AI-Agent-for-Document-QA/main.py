# Dependencies
import os
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Custom modules
from app.ui_helpers import show_pdf_streamlit
from core.rag_pipeline import RAGPipeline
from src.logger import get_logger
from src.custom_exception import CustomException

# vars and logger
load_dotenv()
logger = get_logger(__name__)
vector_db_path = Path("tmp/vector_store")
vector_db_path.mkdir(parents=True, exist_ok=True)

# Modern, sleek header with clear separation
st.markdown("""
<style>
    html, body, .stApp {
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif !important;
        background: #1a1333 !important;
    }
    .main-title {
        font-size: 3.2rem;
        font-weight: 900;
        color: #a78bfa;
        letter-spacing: -1.5px;
        margin-bottom: 0.1em;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        text-shadow: 0 2px 12px #2d1e4a;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #c4b5fd;
        margin-bottom: 1.5em;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .card {
        background: #2d1e4a;
        border-radius: 1.5rem;
        box-shadow: 0 6px 32px #0005;
        padding: 2.5rem 2.2rem 2rem 2.2rem;
        margin-bottom: 2.5rem;
        border: 2px solid #a78bfa;
        transition: box-shadow 0.2s;
    }
    .card:hover {
        box-shadow: 0 12px 40px #a78bfa44;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #c4b5fd;
        margin-bottom: 1.1em;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        letter-spacing: 0.5px;
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 0.7em;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .upload-label {
        font-size: 1.1rem;
        color: #fbbf24;
        font-weight: 600;
        margin-bottom: 0.5em;
        display: block;
    }
    .answer-box {
        margin-top:0.7em;
        font-size:1.13rem;
        color:#f3e8ff;
        background: linear-gradient(90deg, #2d1e4a 60%, #1a1333 100%);
        border-radius:0.9em;
        padding:1.2em 1.4em;
        border:2px solid #a78bfa;
        font-weight: 500;
        box-shadow: 0 2px 8px #0008;
    }
    .source-snippet {
        background: #21173a;
        border-left: 5px solid #a78bfa;
        color: #f3e8ff;
        padding: 0.9em 1.2em;
        border-radius: 0.7em;
        margin-bottom: 0.9em;
        font-size: 1.07rem;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .stButton>button {
        background: #a78bfa;
        color: #1a1333;
        border-radius: 0.7em;
        font-weight: 700;
        font-size: 1.08rem;
        border: none;
        padding: 0.6em 1.5em;
        margin-top: 0.5em;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: #fbbf24;
        color: #2d1e4a;
    }
    .stTextInput>div>input, .stSelectbox>div>div>div>input {
        background: #21173a;
        border-radius: 0.5em;
        border: 1.5px solid #a78bfa;
        font-size: 1.08rem;
        color: #f3e8ff;
        padding: 0.5em 1em;
    }
</style>
<div style='display: flex; align-items: center; gap: 1.5rem; margin-bottom: 0.5em;'>
    <span style='font-size: 3.2rem;'>📄🤖</span>
    <span class='main-title'>DocsQA</span>
</div>
<div class='subtitle'>Your Modern AI Assistant for Document Q&A, Summarization, and Source Highlighting</div>
<hr style='border: none; border-top: 2px solid #a78bfa; margin: 1.2em 0 2em 0;'>
""", unsafe_allow_html=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Sidebar enhancements
st.sidebar.markdown("""
<div class='sidebar-title'>DocsQA Menu</div>
<hr style='border: none; border-top: 1.5px solid #a78bfa; margin: 0.5em 0 1em 0;'>
""", unsafe_allow_html=True)

# File uploader in a visually distinct card
with st.container():
    st.markdown("""
    <div class='card'>
        <span class='upload-label'>Upload a PDF or DOCX</span>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf", "docx"])

if uploaded_file:
    st.sidebar.subheader(uploaded_file.name)
    base_name = uploaded_file.name.rsplit(".", 1)[0]
    temp_path = vector_db_path / f"temp_{uploaded_file.name}"
    document_binary = uploaded_file.read()

    with open(temp_path, "wb") as f:
        f.write(document_binary)

    if uploaded_file.name.endswith(".pdf"):
        show_pdf_streamlit(temp_path, base_name)

    final_path = vector_db_path / base_name / f"{base_name}.{uploaded_file.name.split('.')[-1]}"
    pre_vector_db_path = vector_db_path / base_name / "db"

    if pre_vector_db_path.exists():
        vector_store = Chroma(persist_directory=str(pre_vector_db_path), embedding_function=embedding_model)
        pipeline = RAGPipeline(
            file_path=final_path,
            filename=base_name,
            embedding_model=embedding_model,
            vector_db_path=vector_db_path,
            api_key=os.getenv("OPENROUTER_API_KEY") 
        )
        st.session_state.pipeline = pipeline
        st.session_state.vector_store = vector_store
        if "summary" in st.session_state:
            st.markdown("""
            <div class='card'>
                <span class='section-title'>📝 Document Summary</span>
                <div style='margin-top: 0.7em;'>
                    <div style='color:#f3e8ff; font-size:1.08rem; font-weight:500;'>
                        {summary}
                    </div>
                </div>
            </div>
            """.replace('{summary}', st.session_state.summary), unsafe_allow_html=True)

    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            try:
                with open(final_path, "wb") as f:
                    f.write(document_binary)

                pipeline = RAGPipeline(
                    file_path=final_path,
                    filename=base_name,
                    embedding_model=embedding_model,
                    vector_db_path=vector_db_path,
                    api_key=None
                )
                vector_store = pipeline.build_vector_store()
                summary = pipeline.summarize()

                st.session_state.pipeline = pipeline
                st.session_state.vector_store = vector_store
                st.session_state.summary = summary

                st.markdown("""
                <div class='card'>
                    <span class='section-title'>📄 Document processed successfully.</span>
                    <span class='section-title'>📝 Document Summary</span>
                    <div style='margin-top: 0.7em;'>
                        <div style='color:#f3e8ff; font-size:1.08rem; font-weight:500;'>
                            {summary}
                        </div>
                    </div>
                </div>
                """.replace('{summary}', summary), unsafe_allow_html=True)
                Path(temp_path).unlink()

            except Exception as e:
                logger.error(f"Error while processing document in Streamlit: {e}")
                import sys
                raise CustomException("Error while processing document in Streamlit", sys)

# Assistant and LLM selection in a card
with st.container():
    st.markdown("""
    <div class='card'>
        <span class='section-title'>🤖 Assistant & Model Selection</span>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        assistant = st.text_input("What kind of assistant are you looking for?", placeholder="e.g., Finance")
    with col2:
        LLM_options = [
            "deepseek/deepseek-chat-v3-0324:free",
            "meta-llama/llama-3.3-8b-instruct:free",
            "qwen/qwen3-0.6b-04-28:free",
            "meta-llama/llama-4-maverick:free"
        ]
        LLM_used = st.selectbox("Which LLM do you want to use?", LLM_options)
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)

# Q&A Section in a card
with st.container():
    st.markdown("""
    <div class='card'>
        <span class='section-title'>💬 Ask a Question</span>
    """, unsafe_allow_html=True)
    question = st.text_input("Enter your question:", placeholder="e.g., What is the company's revenue?")
    if st.button("Submit Question") and question:
        with st.spinner("Answering..."):
            try:
                if "pipeline" not in st.session_state or "vector_store" not in st.session_state:
                    st.error("❗ Please process or load a document first.")
                else:
                    pipeline = st.session_state.pipeline
                    vector_store = st.session_state.vector_store
                    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
                    docs = retriever.get_relevant_documents(question)

                    st.markdown("""
                    <div style='margin-bottom: 1.2em;'><span class='section-title'>📌 Source Snippets</span></div>
                    """, unsafe_allow_html=True)
                    for doc in docs:
                        st.markdown(f"<div class='source-snippet'>{doc.page_content[:500]}</div>", unsafe_allow_html=True)
                    st.markdown("<div style='margin-top: 1.2em; font-size: 1.15rem; font-weight: 600; color: #a78bfa;'>🟢 <u>Answer</u></div>", unsafe_allow_html=True)

                    chain = pipeline.get_chain(retriever, assistant, LLM_used)
                    response_placeholder = st.empty()
                    answer = ""
                    for chunk in chain.stream(question):
                        answer += chunk
                        escaped_answer = answer.replace('$', '\\$')
                        response_placeholder.markdown(f"<div class='answer-box'>{escaped_answer}</div>", unsafe_allow_html=True)

            except Exception as e:
                logger.error(f"Error while answering question: {e}")
                import sys
                raise CustomException("Error while answering question in Streamlit", sys)
    st.markdown("</div>", unsafe_allow_html=True)

# About section at the bottom
st.markdown("<div id='about'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='margin-top:2em; text-align:center; color:#a78bfa; font-size:1.1em;'>
    <b>DocsQA</b> &copy; 2024 &mdash; Modern Document Q&A Platform<br>
    <span style='color:#fbbf24;'>Built with ❤️ using Streamlit, LangChain, and HuggingFace</span>
</div>
""", unsafe_allow_html=True)
