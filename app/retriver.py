import os
from dotenv import load_dotenv
from typing import List

# -------------------------------
# Reduce memory pressure (Windows)
# -------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.storage import LocalFileStore, create_kv_docstore

# BUFFER MEMORY
from langchain.memory import ConversationBufferMemory

load_dotenv(override=True)

# ------------------------------------------------
# GLOBAL BUFFER MEMORY
# ------------------------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# ------------------------------------------------
# Load Retriever
# ------------------------------------------------
def load_retriever() -> ParentDocumentRetriever:
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    vectorstore = Chroma(
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "local_hf_pdr"),
        embedding_function=embeddings,
        persist_directory=os.getenv(
            "PERSIST_DIRECTORY", "./data/chroma_db"
        ),
    )

    file_store = LocalFileStore("./data/parent_docs")
    docstore = create_kv_docstore(file_store)

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        ),
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        ),
    )


# ------------------------------------------------
# Helper: format retrieved docs
# ------------------------------------------------
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ------------------------------------------------
# Extract sources
# ------------------------------------------------
def extract_sources(docs: List[Document]) -> List[dict]:
    """
    Extracts human-readable source information
    for frontend display or logging.
    """
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        sources.append(
            {
                "source": os.path.basename(meta.get("source", "unknown")),
                "page": meta.get("page"),
                "snippet": doc.page_content[:300] + "...",
            }
        )
    return sources


# ------------------------------------------------
# Build OpenAI RAG chain WITH MEMORY
# ------------------------------------------------
def build_rag_chain(retriever: ParentDocumentRetriever):
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a careful Retrieval-Augmented Generation (RAG) assistant.

You must follow these rules strictly:

1. Answer ONLY using the provided document context.
2. If multiple documents provide conflicting or contradictory information:
   - Explicitly mention the disagreement.
   - State which document says what.
   - Conclude that the answer is ambiguous or disputed.
3. Do NOT choose one answer arbitrarily when contradictions exist.
4. Only say that you cannot find relevant information if the provided context is clearly unrelated
   or effectively empty. If the context is partially relevant, give the best grounded answer you can
   and explicitly mention any gaps or uncertainty.

---

Chat History:
{chat_history}

---

Context:
{context}

---

Question:
{question}

---

Answer:
"""
    )

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: memory.load_memory_variables({})["chat_history"],
        }
        | prompt
        | llm
    )


# ------------------------------------------------
# Local test run ONLY
# ------------------------------------------------
if __name__ == "__main__":
    print("retriver module running in standalone mode")

    retriever = load_retriever()
    rag_chain = build_rag_chain(retriever)

    query1 = "Explain about the Deepseek-R1-zeromodel"
    response1 = rag_chain.invoke(query1)
    print("\nAnswer 1:\n", response1.content)

    # query2 = "Explain more about greenhouse gases"
    # response2 = rag_chain.invoke(query2)
    # print("\nAnswer 2:\n", response2.content)
