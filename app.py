"""
Prompt-Based RAG Agent — Streamlit Community Cloud entry point.

Env vars are loaded from (in priority order):
  1. Streamlit secrets  (st.secrets)
  2. .env file          (python-dotenv)
  3. Real environment variables
"""

import os
import uuid
import streamlit as st

# ── 0. Page config — must be the very first Streamlit call ───────────────────
st.set_page_config(page_title="Prompt-based RAG Agent", page_icon="📚")

# ── 1. Load configuration before importing the agent ────────────────────────

def _bootstrap_env() -> None:
    """Populate os.environ from Streamlit secrets or .env, whichever is present."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass

    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass


_bootstrap_env()

# ── 2. Import agent (env vars must be set first) ─────────────────────────────

from PromptBasedRagAgent import graph  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402

# ── 3. Helpers ────────────────────────────────────────────────────────────────

def make_thread_id(seed: str) -> str:
    """Return a deterministic UUID5 from *seed* (session key)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def run_graph(messages: list, thread_id: str) -> str:
    """Invoke the LangGraph agent and return the last AI message content."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": messages}, config=config)
    last = result["messages"][-1]
    if hasattr(last, "content"):
        return last.content
    return str(last.get("content", last))


# ── 4. Streamlit UI ───────────────────────────────────────────────────────────

st.title("📚 Prompt-based RAG Agent")

# ── Session seed (stable per browser session) ────────────────────────────────
if "session_seed" not in st.session_state:
    st.session_state.session_seed = str(uuid.uuid4())

thread_id = make_thread_id(st.session_state.session_seed)

# ── Sidebar: thread info + clear ─────────────────────────────────────────────
with st.sidebar:
    st.caption(f"Thread ID: `{thread_id}`")
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# ── Conversation state ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── New user input ────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    lc_messages = []
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = run_graph(lc_messages, thread_id)
            except Exception as exc:
                response = f"⚠️ Error: {exc}"
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})