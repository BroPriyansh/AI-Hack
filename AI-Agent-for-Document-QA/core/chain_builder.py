from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


def build_rag_chain(retriever, assistant, model, api_key):
    try:
        logger.info("Creating RAG chain")
        prompt = """You are an assistant for {assistant}. Use the retrieved context to answer questions. 
                    If you don't know the answer, say so.
                    Always answer in a professional tone.
                    Question: {question}
                    Context: {context}
                    Answer:"""

        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            model=model
        )

        prompt_template = ChatPromptTemplate.from_template(prompt)

        answer= (
            {
                "assistant": RunnableLambda(lambda _: assistant),
                "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain created SUCCESFULLY")
        return answer
    except Exception as e:
            logger.error(f"Error using RAG Chain: {e}")
            raise CustomException("Error using RAG Chain",e)