from core.document_loader import load_and_split_document
from core.vector_store import create_or_load_vectorstore
from core.chain_builder import build_rag_chain
from core.summarizer import summarize_markdown 
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self, file_path, filename, embedding_model, vector_db_path, api_key=None):
        self.file_path = file_path
        self.filename = filename
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        self.api_key = api_key 

    def build_vector_store(self):
        try:
            logger.info("################# Creating Vector Store Pipeline ######################")
            chunks = load_and_split_document(self.file_path)
            vector_store = create_or_load_vectorstore(
                chunks,
                self.filename,
                self.vector_db_path,
                self.embedding_model
            )
            logger.info("################# Vector Store created successfully ###############")
            return vector_store
        except Exception as e:
            logger.error(f"Error in Vector Store Pipeline: {e}")
            import sys
            raise CustomException("Error in Vector Store Pipeline", sys)

    def summarize(self):
        try:
            logger.info("################# Summarizing ######################")
            chunks = load_and_split_document(self.file_path)
            markdown = "\n\n".join(doc.page_content for doc in chunks)
            summary = summarize_markdown(markdown)
            logger.info("################# Summarizing SUCCESSFULLY ######################")
            return summary
        except Exception as e:
            logger.error(f"Error in summarizing: {e}")
            import sys
            raise CustomException("Error in summarizing", sys)


    def get_chain(self, retriever, assistant, model):
        return build_rag_chain(retriever, assistant, model, self.api_key)
