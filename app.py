import streamlit as st
import os
from dotenv import load_dotenv

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store
from langchain_community.vectorstores import FAISS

# HuggingFace Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# =========================
# STREAMLIT SETTINGS
# =========================

st.set_page_config(page_title="RAG App")

st.title("📚 RAG Based Streamlit Application")

st.write("Upload a PDF and ask questions from it.")

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# =========================
# PROCESS PDF
# =========================

if uploaded_file:

    # Save uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF Uploaded Successfully")

    try:

        # =========================
        # LOAD PDF
        # =========================

        loader = PyPDFLoader("temp.pdf")

        documents = loader.load()

        if not documents:
            st.error("❌ No text found in PDF")
            st.stop()

        # =========================
        # TEXT CHUNKING
        # =========================

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            st.error("❌ No chunks created")
            st.stop()

        st.success(f"✅ Chunks Created: {len(chunks)}")

        # =========================
        # HUGGINGFACE EMBEDDINGS
        # =========================

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # =========================
        # VECTOR DATABASE
        # =========================

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        st.success("✅ Vector Database Ready")

        # =========================
        # USER QUESTION
        # =========================

        user_question = st.text_input(
            "Ask Question From PDF"
        )

        if user_question:

            # =========================
            # SIMILARITY SEARCH
            # =========================

            docs = vector_store.similarity_search(
                user_question
            )

            # Create Context
            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            # =========================
            # GEMINI MODEL
            # =========================

            llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=api_key
)

            # Prompt
            prompt = f"""
            Answer the question using the context below.

            Context:
            {context}

            Question:
            {user_question}
            """

            # Generate Response
            response = llm.invoke(prompt)

            # =========================
            # OUTPUT
            # =========================

            st.subheader("📌 Answer")

            st.write(response.content)

    except Exception as e:

        st.error(f"❌ Error: {str(e)}")
