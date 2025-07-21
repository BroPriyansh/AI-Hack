import os
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
import streamlit as st
from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

vector_db_path = Path("tmp") / "vector_store"
vector_db_path.mkdir(parents=True, exist_ok=True)

def show_pdf_streamlit(pdf_path, file_name, folder_override=None):
    try:
        logger.info("Get images from PDF to show in Streamlit")
        
        images_folder = folder_override or (vector_db_path / file_name / "images")
        os.makedirs(images_folder, exist_ok=True)

        image_paths = list(images_folder.glob("*.png"))

        if not image_paths:
            logger.info("Images do not exist, converting PDF to images")
            try:
                doc = fitz.open(pdf_path)
                for i, page in enumerate(doc):
                    pix = page.get_pixmap()
                    img_path = images_folder / f"page_{i + 1}.png"
                    pix.save(str(img_path))
                    image = Image.open(img_path)
                    st.sidebar.image(image, caption=f"Page {i + 1}", use_container_width=True)
                logger.info("Converting PDF to images successfully")
            except Exception as e:
                logger.error(f"Error while converting PDF to images: {e}")
                raise CustomException("Error while converting PDF to images", e)
        else:
            logger.info("Images already exist")
            try:
                for img_path in sorted(image_paths):
                    image = Image.open(img_path)
                    st.sidebar.image(image, caption=f"Page {image_paths.index(img_path) + 1}", use_container_width=True)
                logger.info("Loading existing images successfully")
            except Exception as e:
                logger.error(f"Error while opening existing images: {e}")
                raise CustomException("Error while opening existing images", e)

    except Exception as e:
        logger.error(f"Error while showing PDF in Streamlit: {e}")
        raise CustomException("Error while showing PDF in Streamlit", e)
