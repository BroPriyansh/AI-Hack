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

# App Title
st.title("Smart AI Agent for Document Q&A with Real-Time Summarization and Source Highlighting")

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en", model_kwargs={"device": "cpu"})

uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf"])
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
        st.success("✅ Document already processed. Ready to query.")
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
            st.markdown("### 📝 Document Summary")
            st.success(st.session_state.summary)

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

                st.success("📄 Document processed successfully.")
                st.markdown("### 📝 Document Summary")
                st.success(summary)
                Path(temp_path).unlink()

            except Exception as e:
                logger.error(f"Error while processing document in Streamlit: {e}")
                raise CustomException("Error while processing document in Streamlit", e)

assistant = st.text_input("What kind of assistant are you looking for?", placeholder="e.g., Finance")
LLM_options = [
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "qwen/qwen3-0.6b-04-28:free",
    "meta-llama/llama-4-maverick:free"
]
LLM_used = st.selectbox("Which LLM do you want to use?", LLM_options)
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

                st.markdown("### 📌 Source Snippets")
                for doc in docs:
                    st.info(doc.page_content[:500])
                
                st.markdown("## Answer")

                chain = pipeline.get_chain(retriever, assistant, LLM_used)
                response_placeholder = st.empty()
                answer = ""
                
                for chunk in chain.stream(question):
                    answer += chunk
                    response_placeholder.markdown(answer.replace('$', '\\$'))

        except Exception as e:
            logger.error(f"Error while answering question: {e}")
            raise CustomException("Error while answering question in Streamlit", e)
