import getpass
import os


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=(urls),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

user_query = "agent memory"
joke_query = "Tell me a joke."

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
    Answer the user query.\n{query}\n
    show me output as json object. json object looks like this.
    if retrieval chunk is relavant to query,
    {'relevance': 'yes'}
    if not, 
    {'relevance': 'no'} 
    """,
    input_variables=["relevance", "query"],
    relevance=["yes", "no"],
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | parser

result = rag_chain.invoke({"query": user_query})

print(result)