#  LLM Text Summarizer

An **AI-driven document summarization app** built using **LangChain** and **Hugging Face Transformers**.  
It generates concise, context-aware summaries for long-form documents such as research papers or reports, directly from uploaded PDFs.


---

## 🚀 Features
- 🔹 **PDF Upload & Parsing:** Upload any PDF and extract its full text for analysis.  
- 🔹 **Text Chunking:** Automatically splits long texts into manageable chunks using LangChain’s `RecursiveCharacterTextSplitter`.  
- 🔹 **Embeddings:** Creates semantic embeddings for each text chunk using `SentenceTransformer` (`all-MiniLM-L6-v2`).  
- 🔹 **Summarization Pipeline:** Generates coherent and abstractive summaries using the **LaMini-Flan-T5-248M** model from Hugging Face.  
- 🔹 **Streamlit Interface:** Interactive web UI for easy PDF viewing and summary generation.

---

## 🧩 Tech Stack
- **Python 3.10+**
- **Streamlit** – Web application framework  
- **LangChain** – Text splitting and document handling  
- **Hugging Face Transformers** – Summarization model (T5)  
- **Sentence Transformers** – Embedding generation  
- **PyTorch** – Model backend  
- **PyPDF** – PDF parsing  

---

## 🛠️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sravyamangipudi11/llm-text-summarizer.git
cd llm-text-summarizer
