import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(
    page_title="RAG Book Assistant"
)

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader(
    "Upload a PDF book",
    type="pdf"
)


# -----------------------------
# PDF UPLOAD
# -----------------------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")


    # -----------------------------
    # CREATE VECTOR DATABASE
    # -----------------------------

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            loader = PyPDFLoader(file_path)

            docs = loader.load()


            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)


            # OpenAI Embeddings
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )


            # Chroma Vector Database
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db"
            )


        st.success("Vector database created successfully!")


# -----------------------------
# LOAD VECTOR DATABASE
# -----------------------------

if os.path.exists("chroma_db"):

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )


    # -----------------------------
    # RETRIEVER
    # -----------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


    # -----------------------------
    # MISTRAL MODEL
    # -----------------------------

    llm = ChatMistralAI(
        model="open-mistral-nemo"
    )


    # -----------------------------
    # PROMPT
    # -----------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

ALWAYS tell the page number from which the answer
was found.

If the answer is not present in the context,
say:

"I could not find the answer in the document."
"""
            ),
            (
                "human",
                """
Context:

{context}

Question:

{question}
"""
            )
        ]
    )


    # -----------------------------
    # QUESTION UI
    # -----------------------------

    st.divider()

    st.subheader("Ask Questions From the Book")

    query = st.text_input(
        "Enter your question"
    )


    if query:

        with st.spinner("Searching the book..."):

            # Retrieve documents
            docs = retriever.invoke(query)


            # Context with page numbers
            context = "\n\n".join(
                [
                    f"Page {doc.metadata.get('page', 0) + 1}:\n"
                    f"{doc.page_content}"
                    for doc in docs
                ]
            )


            # Create final prompt
            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )


            # Generate answer
            response = llm.invoke(
                final_prompt
            )


        # -----------------------------
        # ANSWER
        # -----------------------------

        st.write("### 🤖 AI Answer")

        st.write(response.content)


        # -----------------------------
        # SOURCES
        # -----------------------------

        st.write("### 📖 Sources")

        for i, doc in enumerate(docs):

            page_number = (
                doc.metadata.get("page", 0) + 1
            )

            st.write(
                f"**Source {i + 1} — Page {page_number}**"
            )

            st.write(
                doc.page_content[:300] + "..."
            )