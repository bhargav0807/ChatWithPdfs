import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Configure Google Generative AI with API key
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    pdf_texts = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.name  # Get the file name
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        pdf_texts.append((pdf_name, text))  # Store the file name along with the text
    return pdf_texts

def get_text_chunks(pdf_texts):
    """Split text from PDFs into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks_with_metadata = []
    for pdf_name, text in pdf_texts:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_metadata.append({"text": chunk, "source": pdf_name})
    return chunks_with_metadata

def get_vector_store(chunks_with_metadata):
    """Create a vector store from text chunks and save locally."""
    texts = [chunk["text"] for chunk in chunks_with_metadata]
    metadata = [{"source": chunk["source"]} for chunk in chunks_with_metadata]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadata)
    vector_store.save_local("vecDB_metadata")

def get_conversational_chain():
    """Set up a conversational chain with a prompt template and model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, just say, "answer is not available in the context."
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Process user question and display response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("vecDB_metadata", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply: ", response["output_text"])
    st.write("Source PDF(s):")
    for doc in docs:
        st.write(doc.metadata["source"])

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with Multiple PDFs")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete")

if __name__ == "__main__":
    main()
  
