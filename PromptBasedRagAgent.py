import os
import datetime
from pathlib import Path
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

# ── Document loaders ──────────────────────────────────────────────────────────
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Local embeddings + vector store ──────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PROMPT_NAME  = "agent.prompt"
PROMPT_PATH  = os.path.join(os.path.dirname(__file__), "prompts", PROMPT_NAME)
RAG_DIR      = os.path.join(os.path.dirname(__file__), "rag")
OPENAI_MODEL = "gpt-4.1-mini"

# Local embedding model – downloaded once, cached in ~/.cache/huggingface
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Build RAG index at startup ────────────────────────────────────────────────

_LOADERS = {
    ".txt":  TextLoader,
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
}

def _load_documents():
    """Load all supported files from the rag/ folder."""
    docs = []
    rag_path = Path(RAG_DIR)
    for path in rag_path.iterdir():
        loader_cls = _LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        try:
            loader = loader_cls(str(path))
            loaded = loader.load()
            # Tag each chunk with its source filename
            for doc in loaded:
                doc.metadata.setdefault("source", path.name)
            docs.extend(loaded)
            print(f"[RAG] Loaded: {path.name} ({len(loaded)} chunk(s))")
        except Exception as e:
            print(f"[RAG] Warning — could not load {path.name}: {e}")
    return docs


def _build_index():
    """Return a FAISS retriever, or None if the rag/ folder is empty."""
    docs = _load_documents()
    if not docs:
        print("[RAG] No documents found in rag/ — retrieval tool will be disabled.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    print(f"[RAG] Index built: {len(chunks)} chunks from {len(docs)} document(s).")
    return db.as_retriever(search_kwargs={"k": 4})


_retriever = _build_index()

# ── Tools ─────────────────────────────────────────────────────────────────────

def get_current_date() -> str:
    """Get today's date in ISO format."""
    return datetime.date.today().isoformat()


def search_documents(query: str) -> str:
    """Search the internal knowledge base for information relevant to the query.
    Returns the most relevant passages found in the loaded documents."""
    if _retriever is None:
        return "No documents are available in the knowledge base."
    try:
        results = _retriever.invoke(query)
        if not results:
            return "No relevant passages found."
        parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Error during document search: {e}"


# ── Prompt ────────────────────────────────────────────────────────────────────

def _load_system_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

base_system_prompt = _load_system_prompt()


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    system_msg = base_system_prompt
    return [{"role": "system", "content": system_msg}] + state["messages"]


# ── Graph ─────────────────────────────────────────────────────────────────────

_tools = [get_current_date, search_documents]

graph = create_react_agent(
    model=f"openai:{OPENAI_MODEL}",
    tools=_tools,
    prompt=prompt,
)