# 📄 LangChain PDF Q&A Agent

Ask natural language questions about any PDF document.
Built with LangChain, Groq LLaMA 3, FAISS, and Streamlit.

## How it works
1. Upload any PDF
2. Document is split into chunks and embedded
3. Top 3 relevant chunks retrieved per question
4. Groq LLaMA 3 answers based on those chunks

## Tech Stack
- LangChain — retrieval orchestration
- Groq API — free LLM inference (LLaMA 3.3 70B)
- FAISS — vector store for semantic search
- Streamlit — web UI

## Run locally
git clone https://github.com/vishu389/doc-qa-agent
cd doc-qa-agent
pip install -r requirements.txt
streamlit run app.py
