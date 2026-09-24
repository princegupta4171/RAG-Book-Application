
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

data = PyPDFLoader("document loaders/deeplearning.pdf")

docs = data.load()

spilitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)


chunks = spilitter.split_documents(docs)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)