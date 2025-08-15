import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
import os
from dotenv import load_dotenv

load_dotenv()


DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'YOUR_DEEPSEEK_API_KEY')  # Set your key here
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")
retriever = db.as_retriever(search_kwargs={"k":3})
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

def ask_deepseek(question, context):
    prompt = f"Answer the following question using ONLY the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nIf you use information from the context, cite the source."
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} {response.text}"

st.title("RAG PDF QA with DeepSeek LLM")

query = st.text_input("Ask a question about your documents:")

if query:
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [f"{doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', 'N/A')})" for doc in docs]
    answer = ask_deepseek(query, context)

    st.markdown(f"**Answer:** {answer}")
    st.markdown("**Sources:**")
    for src in sources:
        st.markdown(f"- {src}")
    with st.expander("Show Source Texts"):
        for doc in docs:
            st.write(doc.page_content)