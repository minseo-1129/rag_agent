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
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Local embeddings + vector store ──────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Web search ────────────────────────────────────────────────────────────────
from langchain_community.tools import DuckDuckGoSearchRun


PROMPT_NAME  = "agent.prompt"
PROMPT_PATH  = os.path.join(os.path.dirname(__file__), "prompts", PROMPT_NAME)
RAG_DIR      = os.path.join(os.path.dirname(__file__), "rag")
OPENAI_MODEL = "gpt-4o-mini"

# Local embedding model – downloaded once, cached in ~/.cache/huggingface
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── URLs to index in RAG ──────────────────────────────────────────────────────
RESOURCE_URLS = [
    {
        "url": "https://www.lbhf.gov.uk/health-and-care/reach-out-help-available?gad_source=1&gad_campaignid=22048830849",
        "description": "LBHF Reach Out - Local mental health and wellbeing support services"
    },
    {
        "url": "https://www.westlondon.nhs.uk/our-services/adult/mental-health-services/Employment-Support-Services-Individual-Placement-and-Support/employment-support-services-your-community",
        "description": "Support services and resources in West London Trust"
    },
    # Add more URLs here as needed:
    # {
    #     "url": "https://www.nhs.uk/mental-health/",
    #     "description": "NHS Mental Health Services"
    # },
]

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
    
    if not rag_path.exists():
        print(f"[RAG] Warning: {RAG_DIR} folder not found.")
        return docs
    
    for path in rag_path.iterdir():
        if not path.is_file():
            continue
        loader_cls = _LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        try:
            loader = loader_cls(str(path))
            loaded = loader.load()
            # Tag each chunk with its source filename
            for doc in loaded:
                doc.metadata.setdefault("source", path.name)
                doc.metadata["type"] = "local_file"
            docs.extend(loaded)
            print(f"[RAG] Loaded: {path.name} ({len(loaded)} chunk(s))")
        except Exception as e:
            print(f"[RAG] Warning — could not load {path.name}: {e}")
    return docs


def _load_url_documents():
    """Load documents from configured URLs."""
    docs = []
    for resource in RESOURCE_URLS:
        try:
            print(f"[RAG] Fetching URL: {resource['description']}...")
            loader = WebBaseLoader(resource["url"])
            loaded = loader.load()
            # Tag with URL and description
            for doc in loaded:
                doc.metadata["source"] = resource["url"]
                doc.metadata["description"] = resource["description"]
                doc.metadata["type"] = "web_resource"
            docs.extend(loaded)
            print(f"[RAG] Loaded URL: {resource['description']} ({len(loaded)} chunk(s))")
        except Exception as e:
            print(f"[RAG] Warning — could not load {resource['url']}: {e}")
    return docs


def _build_index():
    """Return a FAISS retriever with both local files and URLs, or None if empty."""
    # Load local documents
    local_docs = _load_documents()
    
    # Load URL documents
    url_docs = _load_url_documents()
    
    # Combine all documents
    all_docs = local_docs + url_docs
    
    if not all_docs:
        print("[RAG] No documents found — retrieval tool will be disabled.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    print(f"[RAG] Index built: {len(chunks)} chunks from {len(all_docs)} document(s) "
          f"({len(local_docs)} local, {len(url_docs)} web)")
    return db.as_retriever(search_kwargs={"k": 4})


print("[RAG] Building index...")
_retriever = _build_index()

# ── Tools ─────────────────────────────────────────────────────────────────────

def get_current_date() -> str:
    """Get today's date in ISO format."""
    return datetime.date.today().isoformat()


def search_documents(query: str) -> str:
    """Search the internal knowledge base for information relevant to the query.
    This searches both local documents AND trusted web resources that have been indexed.
    Returns the most relevant passages including URLs for external resources.
    
    Use this tool to find:
    - Self-help resources and coping strategies
    - Mental health support services and helplines
    - Professional resources and guidance
    """
    if _retriever is None:
        return "No documents are available in the knowledge base."
    try:
        results = _retriever.invoke(query)
        if not results:
            return "No relevant passages found."
        
        parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            doc_type = doc.metadata.get("type", "unknown")
            description = doc.metadata.get("description", "")
            
            # Format output differently for URLs vs files
            if doc_type == "web_resource":
                header = f"[{i}] {description}\nURL: {source}"
            else:
                header = f"[{i}] Source: {source}"
            
            parts.append(f"{header}\n{doc.page_content.strip()}")
        
        return "\n\n".join(parts)
    except Exception as e:
        return f"Error during document search: {e}"


def web_search(query: str) -> str:
    """Search the web for very recent or current information not available in the knowledge base.
    
    Examples of when to use this:
    - Current news or events (e.g., "latest mental health research 2024")
    - Real-time information (e.g., "current wait times for NHS mental health services")
    - Recently updated guidelines or policies
    
    Do NOT use this for:
    - General mental health information (use search_documents instead)
    - Coping strategies or self-help resources (use search_documents instead)
    - Support services and helplines (use search_documents instead)
    """
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results
    except Exception as e:
        return f"Error during web search: {e}"


# ── Prompt ────────────────────────────────────────────────────────────────────

def _load_system_prompt() -> str:
    """Load the base system prompt from file."""
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"[RAG] Warning: {PROMPT_PATH} not found. Using minimal prompt.")
        return "You are a helpful mental health support assistant."


base_system_prompt = _load_system_prompt()


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    """Inject system prompt with tool usage guidelines."""
    
    tool_guidelines = """

TOOL USAGE GUIDELINES:

PRIMARY TOOL - search_documents:
- Use this as your FIRST and MAIN tool for ALL mental health related queries
- This searches both local documents AND trusted web resources (already indexed)
- Use it for: coping strategies, support services, helplines, self-help resources, professional guidance

SECONDARY TOOL - web_search:
- ONLY use this in rare cases when you need VERY recent information (last few weeks/months)
- Examples: "latest mental health statistics 2024", "new NHS policy announced this month"
- Do NOT use for general mental health information - that's in search_documents
- If you're unsure, try search_documents first

OTHER TOOLS:
- get_current_date: Use when date/time information is relevant to the conversation

RESOURCE OFFERING PROTOCOL:
If the user shows resistance to feeling emotions for more than 3 conversational turns:
1. First ask permission: "Would it be okay if I share some resources that might help manage what you're experiencing?"
2. If they agree, use 'search_documents' to find appropriate resources
3. Offer only ONE suggestion at a time
4. ALWAYS cite the source with full details:
   - For web resources: Include the description and URL
   - For local files: Include the filename
5. Explain WHY you're suggesting this specific resource based on their situation
6. After sharing the resource, provide a brief summary of the conversation and close

When citing sources:
- Be specific: "According to [source name] at [URL]..." or "From the document [filename]..."
- Make it easy for the user to access the resource
"""
    
    system_msg = base_system_prompt + tool_guidelines
    return [{"role": "system", "content": system_msg}] + state["messages"]


# ── Graph ─────────────────────────────────────────────────────────────────────

_tools = [
    get_current_date,
    search_documents,
    web_search,
]

graph = create_react_agent(
    model=f"openai:{OPENAI_MODEL}",
    tools=_tools,
    prompt=prompt,
)
