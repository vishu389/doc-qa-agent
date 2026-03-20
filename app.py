import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Doc Q&A Agent", page_icon="🤖")
st.title("📄 Document Q&A Agent")
st.caption("Upload any PDF and ask questions — powered by LangChain + Groq")

GROQ_API_KEY = "your-groq-api-key-here"

@st.cache_resource
def build_qa_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0)
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {question}
""")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Reading and indexing document..."):
        qa_chain, retriever = build_qa_chain(temp_path)
    st.success("Ready! Ask anything below.")
    question = st.text_input("Your question:")
    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(question)
            sources = retriever.invoke(question)
        st.markdown("### Answer")
        st.write(answer)
        with st.expander("View source chunks"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Chunk {i+1} — Page {doc.metadata.get('page', '?')+1}**")
                st.caption(doc.page_content[:300])
else:
    st.info("Please upload a PDF to get started.")