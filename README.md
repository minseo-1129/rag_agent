# Prompt-Based RAG Agent

A LangGraph ReAct agent with a local RAG knowledge base.  
Embeddings are computed locally with `sentence-transformers` — **no embedding API cost**.

---

## How RAG works here

```
rag/
  your-doc.pdf
  notes.txt
  manual.docx
```

At startup `PromptBasedRagAgent.py`:
1. Loads every `.txt`, `.pdf`, and `.docx` file from the `rag/` folder.
2. Splits them into 500-token chunks (50-token overlap).
3. Embeds them with `all-MiniLM-L6-v2` (runs locally, ~80 MB, cached after first download).
4. Builds an in-memory **FAISS** index.
5. Exposes a `search_documents(query)` tool the ReAct agent calls when needed.

The index is rebuilt on every cold start. For Streamlit Community Cloud this happens once per deploy/restart.

---

## Repo structure

```
app.py                          ← Streamlit entry point (unchanged pattern)
PromptBasedRagAgent.py          ← LangGraph agent + RAG index builder
prompts/
  agent.prompt                  ← System prompt (edit freely)
rag/
  .gitkeep                      ← Drop your documents here
requirements.txt
.env.example
```

---

## Local development

```bash
pip install -r requirements.txt

cp .env.example .env
# add OPENAI_API_KEY to .env

# Add documents
cp my-docs/*.pdf rag/

streamlit run app.py
```

---

## Streamlit Community Cloud

1. Push repo to GitHub **including your documents in `rag/`**.
2. Create app → select `app.py`.
3. App settings → Secrets:

```toml
OPENAI_API_KEY = "sk-..."
```

> **Note:** The HuggingFace model (~80 MB) is downloaded on first boot and cached.  
> Subsequent restarts reuse the cache and start faster.

---

## Supported document formats

| Extension | Loader |
|---|---|
| `.txt` | `TextLoader` |
| `.pdf` | `PyPDFLoader` |
| `.docx` | `Docx2txtLoader` |

Add new files to `rag/` and redeploy — the index rebuilds automatically. Do not forget to remove the Conan Doyle novel in the folder.