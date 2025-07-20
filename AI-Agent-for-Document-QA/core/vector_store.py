from pathlib import Path
from langchain_community.vectorstores import Chroma
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def create_or_load_vectorstore(chunks, filename, vector_db_path, embedding_model):
    try:
        logger.info(f"Creating Vector Store: {filename}")
        db_dir = Path(vector_db_path) / filename / "db"
        
        if db_dir.exists():
            chroma_db=Chroma(persist_directory=str(db_dir), embedding_function=embedding_model)
        else:
            chroma_db= Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=str(db_dir))
  
        logger.info(f"Creating Vector Store SUCCESSFULLY")
        return chroma_db
    except Exception as e:
            logger.error(f"Error while creating vector store: {e}")
            raise CustomException("Error while creating vector store",e)
