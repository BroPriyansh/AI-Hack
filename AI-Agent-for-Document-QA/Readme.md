# 🤖 Smart AI Agent for Document Q&A with Real-Time Summarization and Source Highlighting

---

## Overview

An end-to-end intelligent document assistant that allows users to upload **PDF** business documents, receive a **real-time summary**, ask **natural language questions**, and view the **source snippets** from which answers are retrieved — with optional highlighting.
---

# 📽️ Demo

📽️ [Demo - YouTube](https://youtu.be/bE4pniYSH5I)

🧪 Try it live on Hugging Face:  

**[👉 Smart AI Agent for Document Q&A (Streamlit App)](https://huggingface.co/spaces/WesleyGonzales/AI-Agent-for-Document-QA)**

![huggingfaces](https://github.com/user-attachments/assets/0eaa6eec-176c-4203-9453-3b86237e8e88)

![huggingfaces_2](https://github.com/user-attachments/assets/356fa60b-0a03-4dd7-b944-c42f02b55cb7)

---

## 🚀 Features

- 📄 Supports **PDF** file formats
- 🧠 Summarizes the document on upload using Hugging Face models
- ❓ Allows users to ask **natural language questions**
- 🧷 Shows relevant **source snippets** per answer
- 💡 Built with **LangChain**, **HuggingFace Transformers**, **ChromaDB**, and **Streamlit**

---

## 📦 Tech Stack

| Component                    | Description                                  |
|------------------------------|----------------------------------------------|
| Python 3.10+                 | Programming language                                    |
| Streamlit                  | Frontend UI framework                                     |
| LangChain                  | RAG (Retrieval-Augmented Generation)          |
| ChromaDB                  | Vector store for document chunks                  |
| HuggingFace               | Embedding + Summarization models                      |
|PyMuPDF                 | Document parsing + highlighting                  |


---

## 📂 Project Structure

```
📁 core/                
│ ├── chain_builder.py # LangChain RAG chain builder
│ ├── document_loader.py # PDF/DOCX text extraction
│ ├── rag_pipeline.py # Main RAG pipeline class
│ ├── splitter.py # Markdown-based text splitting
│ ├── summarizer.py # HF summarization logic
│ └── vector_store.py # Create/load Chroma vector DB
📁 app/
│ └── ui_helpers.py # Streamlit PDF image display
📁 src/
│ ├── custom_exception.py # Custom error handling
│ └── logger.py # Unified logger
📁 tmp/
│ └── vector_store/ # Local persistent vector DBs
├── Dockerfile              # Docker setup for HF Spaces
├── requirements.txt        # Python dependencies
├── README.md               # You're here!
```

---

## 🚀 How to Run (Local PC / Streamlit Local / Streamlit Cloud)

### Local PC

1. Clone the repo
```bash
git clone https://github.com/WesleyG31/AI-Agent-for-Document-QA
cd AI-Agent-for-Document-QA
```

2. (Optional) Create a virtual environment with Anaconda
```bash
conda create -n AI-agent-QA python=3.10
conda activate AI-agent-QA
```

3. Install dependencies
```bash
pip install -e .
```

4. (Optional) Install Cuda
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

5. Add your OpenRouter API key
```bash
Create an file called .env 
OPENROUTER_API_KEY=your-key-here
```

6. Run the python file locally
```bash
streamlit run main.py
```

###  HuggingFace Spaces

1. No installation required. Just open the link and upload a PDF.
```bash
https://huggingface.co/spaces/WesleyGonzales/AI-Agent-for-Document-QA
PD: It's the same if you run it locally
```

---

## 🔒 API & Credentials
This app uses OpenRouter to access LLMs. Create a free account and get your API key.

In production, store the API key securely via Hugging Face Secrets or .env.

---

## 📄 Sample Outputs

- ✅ Answers about your PDF document.
- ✅ Summarizes the document
- ✅ Shows relevant **source snippets** per answer
---

## 💼 Why This Project Matters

This project demonstrates:

- Solid architecture with LangChain, vector stores, and custom RAG

- Practical LLM integration with real-world file formats (PDFs)

- Full deployment with Docker & Hugging Face Spaces

- Thoughtful error handling, logging, and modular code structure


---

## 👨‍💻 Author

**Wesley Gonzales**  
AI & ML Engineer  
📫 wes.gb31@gmail.com  
🔗 [https://www.linkedin.com/in/wesleygb/](https://www.linkedin.com/in/wesleygb/)  
🤖 [My Github](https://github.com/WesleyG31)
---

## 🪪 License

This project is licensed under the MIT License.
