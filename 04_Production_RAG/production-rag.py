#!/usr/bin/env python3
import logging

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import tiktoken

FORMAT = '%(asctime)s %(levelname)s %(message)s'
LOG_SETTINGS = {
    'format': FORMAT,
    'level': logging.INFO,
    'datefmt': "%Y-%m-%dT%H:%MM:%S",
}
logging.basicConfig(**LOG_SETTINGS)
logger = logging.getLogger()

load_dotenv()
openai_chat_model = ChatOpenAI(model="gpt-4o-mini")



def main():
    logger.info(f"{'-' * 20} main")
    system_template = "You are a legendary and mythical Wizard. You speak in riddles and make obscure and pun-filled references to exotic cheeses."
    human_template = "{content}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    chain = chat_prompt | openai_chat_model
    print(chain.invoke({"content": "Hello world!"}))

    print(
        chain.invoke(
            {"content" : "Could I please have some advice on how to become a better Python Programmer?"}
        )
    )

def naive_rag():
    logger.info(f"{'-' * 20} naive_rag")
    system_template = "You are a helpful assistant."
    human_template = "{content}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

    chat_chain = chat_prompt | openai_chat_model ### LCEL Chain!

    args = {"content" : "Please define LangGraph."}
    cc = chat_chain.invoke(args)
    print(cc.content)
    print(cc.response_metadata)

    args = {"content" : "What is LangChain Expression Language (LECL)?"}
    cc = chat_chain.invoke(args)
    print(cc.content)
    print(cc.response_metadata)


HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provide context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context response with "I don't know"
"""

CONTEXT = """
LangChain Expression Language or LCEL is a declarative way to easily compose chains together. There are several benefits to writing chains in this manner (as opposed to writing normal code):

Async, Batch, and Streaming Support Any chain constructed this way will automatically have full sync, async, batch, and streaming support. This makes it easy to prototype a chain in a Jupyter notebook using the sync interface, and then expose it as an async streaming interface.

Fallbacks The non-determinism of LLMs makes it important to be able to handle errors gracefully. With LCEL you can easily attach fallbacks to any chain.

Parallelism Since LLM applications involve (sometimes long) API calls, it often becomes important to run things in parallel. With LCEL syntax, any components that can be run in parallel automatically are.

Seamless LangSmith Tracing Integration As your chains get more and more complex, it becomes increasingly important to understand what exactly is happening at every step. With LCEL, all steps are automatically logged to LangSmith for maximal observability and debuggability.
"""

def naive_rag_with_context():
    logger.info(f"{'-' * 20} naive_rag_with_context")
    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", HUMAN_TEMPLATE)
    ])

    chat_chain = chat_prompt | openai_chat_model

    args = {"query" : "What is LangChain Expression Language?", "context" : CONTEXT}
    cc = chat_chain.invoke(args)
    print(cc.content)
    print(cc.response_metadata)



def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(
        text,
    )
    return len(tokens)

def create_chunks():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 0,
        length_function = tiktoken_len,
    )
    chunks = text_splitter.split_text(CONTEXT)
    return chunks


def explore_chunks():
    logger.info(f"{'-' * 20} explore_chunks")
    chunks = create_chunks()
    print("\n----\n".join(chunks))


def explore_embedding_model():
    logger.info(f"{'-' * 20} explore_embedding_model")
    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", HUMAN_TEMPLATE)
    ])

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_dim = 1536
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="lcel_doc_v1",
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="lcel_doc_v1",
        embedding=embedding_model,
    )

    chunks = create_chunks()
    _ = vector_store.add_texts(texts=chunks)

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    simple_rag  = (
        {"context": retriever, "query": RunnablePassthrough()}
        | chat_prompt
        | openai_chat_model
        | StrOutputParser()
    )

    x = simple_rag.invoke("What is LCEL?")
    print(x)

if __name__ == '__main__':
    # main()
    # naive_rag()
    # naive_rag_with_context()
    # explore_chunks()
    explore_embedding_model()

