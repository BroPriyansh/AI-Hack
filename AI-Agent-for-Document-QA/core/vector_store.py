from langchain_community.vectorstores import FAISS
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def create_or_load_vectorstore(chunks, filename, vector_db_path, embedding_model):
    try:
        logger.info(f"Creating Vector Store (FAISS, in-memory): {filename}")
        # FAISS does not persist to disk by default; this is in-memory only
        faiss_db = FAISS.from_documents(documents=chunks, embedding=embedding_model)
        logger.info(f"Creating Vector Store SUCCESSFULLY (FAISS, in-memory)")
        return faiss_db
    except Exception as e:
        logger.error(f"Error while creating vector store: {e}")
        try:
            import streamlit as st
            st.error(f"❗ Error while creating vector store: {e}")
        except Exception:
            pass
        import sys
        raise CustomException("Error while creating vector store", sys)
