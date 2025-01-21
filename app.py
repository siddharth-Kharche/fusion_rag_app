import os
import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.evals import RagasAnswerRelevancy
from athina.loaders import Loader

from utils import reciprocal_rank_fusion, create_query_generation_chain

# Set Streamlit page configuration
st.set_page_config(page_title="RAG Fusion App", page_icon=":books:")

# --- SIDEBAR ---
with st.sidebar:
    st.title('RAG Fusion App :books:')
    st.header("API Keys")

    # Optional: Provide a warning if keys are not present in secrets
    if "OPENAI_API_KEY" not in st.secrets or "ATHINA_API_KEY" not in st.secrets or "QDRANT_API_KEY" not in st.secrets:
        st.warning("Please make sure you've provided the API keys in st.secrets.")

    openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.secrets.get("OPENAI_API_KEY", ""))
    athina_api_key = st.text_input("Athina API Key", type="password", value=st.secrets.get("ATHINA_API_KEY", ""))
    qdrant_api_key = st.text_input("Qdrant API Key", type="password", value=st.secrets.get("QDRANT_API_KEY", ""))
    qdrant_url = st.text_input("Qdrant URL", value="your_qdrant_url")

    # store keys in env vars
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["ATHINA_API_KEY"] = athina_api_key
    os.environ["QDRANT_API_KEY"] = qdrant_api_key

    if not openai_api_key or not athina_api_key or not qdrant_api_key:
            st.stop() # Stop if keys not provided
# -------------------


# 1. Indexing
@st.cache_resource
def load_and_index_data(qdrant_url):
    embeddings = OpenAIEmbeddings()
    loader = CSVLoader("./data/context.csv")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    url=qdrant_url,
    prefer_grpc=True,
    collection_name="documents",
    api_key=os.environ["QDRANT_API_KEY"],
    )

    retriever = vectorstore.as_retriever()
    return retriever, embeddings
    
retriever, embeddings = load_and_index_data(qdrant_url)

# 2. RRF Chain
generate_queries = create_query_generation_chain("langchain-ai/rag-fusion-query-generation")

def create_chain(retriever, generate_queries):
    chain = generate_queries | retriever.map() | reciprocal_rank_fusion
    return chain

chain = create_chain(retriever, generate_queries)

# 3. RAG Chain
llm = ChatOpenAI()

template = """Answer the question based only on the following context.
If you don't find the answer in the context, just say that you don't know.

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_fusion_chain = (
    {
        "context": chain,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --- MAIN APP ---
st.title("RAG Fusion")

query = st.text_input("Enter your query here")

if query:
    response = rag_fusion_chain.invoke(query)
    st.subheader("Response:")
    st.write(response)

    with st.expander("Context"):
            contexts = [doc.page_content for doc in retriever.get_relevant_documents(query)]
            st.write(contexts)

    ground_truths = st.text_input("Enter your ground truth here (optional)")

    if ground_truths:
          question = [query]
          response = [response]
          contexts = [[doc.page_content for doc in retriever.get_relevant_documents(query)]]
          ground_truths = [ground_truths]

          data = {
                "query": question,
                "response": response,
                "context": contexts,
                "ground_truth": ground_truths
          }

          dataset = Loader().load_dict(data)

          st.subheader("Evaluation:")
          OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
          AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))
          eval_df = RagasAnswerRelevancy(model="gpt-4o").run_batch(data=dataset).to_df()
          st.write(eval_df)