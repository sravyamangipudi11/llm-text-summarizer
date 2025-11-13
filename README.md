
### README.md

# 🧠 LLM Text Summarizer

An **AI-driven document summarization app** built using **LangChain** and **Hugging Face Transformers**.  
It generates concise, context-aware summaries for long-form documents such as research papers or reports, directly from uploaded PDFs.
uals.*

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
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```


---

## ▶️ Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

Open the local URL (usually `http://localhost:8501`) in your browser.

### 🧠 How It Works
1. **Upload a PDF**: The app loads and splits your document into smaller text chunks.
2. **Generate Embeddings**: Each chunk is encoded into embeddings for semantic representation.
3. **Summarize**: The `LaMini-Flan-T5` transformer model creates a readable, context-preserving summary.
4. **Display**: The summary and original PDF are shown side by side on the web interface.

---

## 🧪 Example Output
**Input**:  
“Artificial intelligence (AI) has emerged as a transformative technology across various industries…”

**Output Summary**:  
“AI is revolutionizing multiple sectors through automation, data analysis and predictive modeling, leading to improved efficiency and decision-making.”

---

## 📂 Project Structure
```
📦 llm-text-summarizer
 ┣ 📜 app.py                 # Main Streamlit application
 ┣ 📂 doc/                   # Uploaded PDFs
 ┣ 📜 requirements.txt       # Dependencies
 ┣ 📜 README.md              # Project documentation
```

---

## 📚 Model Info
- **Summarization Model**: `MBZUAI/LaMini-Flan-T5-248M`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

---


---

## 🪪 License
This project is released under the [MIT License](LICENSE).
