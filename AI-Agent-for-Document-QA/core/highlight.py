import fitz
from pathlib import Path
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def highlight_pdf_answer(original_pdf_path, matched_chunks, output_path):
    try:
        logger.info(f"Highlighting answers in PDF: {original_pdf_path}")
        doc = fitz.open(original_pdf_path)

        for chunk in matched_chunks:
            page_num = chunk.metadata.get("page")
            if page_num is None:
                continue  

            try:
                page = doc[int(page_num)]  
                text = chunk.page_content.strip()

                highlights = page.search_for(text, quads=True)
                if not highlights:
                    text_snippet = text[:80]
                    highlights = page.search_for(text_snippet, quads=True)

                for rect in highlights:
                    page.add_highlight_annot(rect)
            except Exception as e:
                logger.warning(f"Could not highlight on page {page_num}: {e}")
                continue

        doc.save(output_path)
        doc.close()
        logger.info(f"Highlighted PDF saved to: {output_path}")
        return Path(output_path)

    except Exception as e:
        logger.error(f"Error while highlighting PDF: {e}")
        raise CustomException("Error while highlighting PDF", e)
