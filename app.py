import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os

import os

# Load model and tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)


if not os.path.exists("doc"):
    os.makedirs("doc")

# File preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# LLM summarization pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
        device=0 if torch.cuda.is_available() else -1
    )
    input_text = file_preprocessing(filepath)
    if len(input_text) > 2000:
        input_text = input_text[:2000]
    result = pipe_sum(input_text)
    return result[0]["summary_text"]

# Function to display PDF
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit app
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        filepath = os.path.join("doc", uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns(2)
        with col1:
            st.info("Uploaded File")
            displayPDF(filepath)

        if st.button("Summarize"):
            with st.spinner("Summarizing... Please wait."):
                summary = llm_pipeline(filepath)
                st.success("Summarization Complete!")
                with col2:
                    st.subheader("Summary")
                    st.write(summary)

if __name__ == "__main__":
    main()
