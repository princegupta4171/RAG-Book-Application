# 📚 RAG Book Application

A Retrieval-Augmented Generation (RAG) system that lets you upload any PDF book and ask questions from it. Built with LangChain, ChromaDB, OpenAI Embeddings, and Mistral AI.

## 🚀 Live Demo

Upload a PDF → Create Vector Database → Ask Questions → Get AI answers with page numbers.

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | Mistral AI (`open-mistral-nemo`) |
| Embeddings | OpenAI (`text-embedding-3-small`) |
| Vector Store | ChromaDB |
| UI | Streamlit |

## 📁 Project Structure

```
RAG-Book-Application/
├── app.py                  # Main Streamlit app
├── main2.py                # CLI-based RAG pipeline
├── create_databse.py       # Script to create ChromaDB from PDF
├── vector_store/
│   └── db.py               # Vector store setup & retrieval demo
├── document_loaders/
│   ├── pdf.py              # PDF loader example
│   ├── page.py             # Web page loader example
│   ├── test.py
│   ├── splitertest.py
│   └── tokensplittertest.py
├── requirements.txt
└── .gitignore
```

## ⚙️ Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/princegupta4171/RAG-Book-Application.git
   cd RAG-Book-Application
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file**
   ```
   OPENAI_API_KEY=<your_openai_api_key>
   MISTRAL_API_KEY=<your_mistral_api_key>
   ```

## ▶️ Run

**Streamlit App (Recommended)**
```bash
streamlit run app.py
```

**CLI Mode**
```bash
python main2.py
```

## 💡 How It Works

1. Upload a PDF via the Streamlit UI
2. PDF is split into chunks using `RecursiveCharacterTextSplitter`
3. Chunks are embedded using OpenAI and stored in ChromaDB
4. On query, MMR retriever fetches the most relevant chunks
5. Mistral AI generates an answer with page number references

## 🔑 API Keys Required

- [OpenAI API Key](https://platform.openai.com/api-keys)
- [Mistral AI API Key](https://console.mistral.ai/)
