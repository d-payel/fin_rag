import streamlit as st
import os
import time
from src.rag_pipeline_2 import FinancialRAG
from src.utils import format_sources, get_sample_questions


from dotenv import load_dotenv
load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinDoc RAG — Financial Report Q&A",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0f1117; }
  .stApp { background: #0f1117; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #161b2e !important;
    border-right: 1px solid #1e2a45;
  }

  /* Header */
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #e8f0fe;
    line-height: 1.2;
    margin-bottom: 4px;
  }
  .hero-sub {
    font-size: 0.9rem;
    color: #6b7db3;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 24px;
  }

  /* Chat messages */
  .msg-user {
    background: #1a2340;
    border: 1px solid #2a3a60;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #c8d8f8;
    font-size: 0.95rem;
    text-align: right;
  }
  .msg-bot {
    background: #111827;
    border: 1px solid #1e2a45;
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #d1ddf5;
    font-size: 0.95rem;
  }
  .msg-bot p { margin: 0 0 8px 0; }

  /* Source cards */
  .source-card {
    background: #0d1526;
    border: 1px solid #1e2d50;
    border-left: 3px solid #3b6fd4;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: #8a9fc7;
  }
  .source-card strong { color: #5b8dee; }

  /* Chips */
  .chip {
    display: inline-block;
    background: #1a2a4a;
    border: 1px solid #2a3f6f;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #7090c8;
    margin: 4px 4px 4px 0;
    cursor: pointer;
  }

  /* Metric cards */
  .metric-card {
    background: #111827;
    border: 1px solid #1e2a45;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }
  .metric-num { font-size: 1.6rem; font-weight: 600; color: #5b8dee; }
  .metric-label { font-size: 0.75rem; color: #6b7db3; text-transform: uppercase; letter-spacing: 0.06em; }

  /* Divider */
  hr { border-color: #1e2a45 !important; }

  /* Input */
  .stTextInput input, .stChatInput textarea {
    background: #161b2e !important;
    border: 1px solid #2a3a60 !important;
    color: #c8d8f8 !important;
  }

  /* Buttons */
  .stButton button {
    background: #1e3a7a;
    color: #c8d8f8;
    border: 1px solid #2a4a9a;
    border-radius: 8px;
    font-size: 0.85rem;
  }
  .stButton button:hover { background: #2a4a9a; border-color: #4a7ae8; }

  /* File uploader */
  .stFileUploader { background: #111827; border-radius: 10px; }

  /* Status badge */
  .badge-ready {
    background: #0d2b1a;
    border: 1px solid #1a5c35;
    color: #4caf82;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    display: inline-block;
  }
  .badge-empty {
    background: #2a1a0d;
    border: 1px solid #5c3a1a;
    color: #c8834f;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    display: inline-block;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">FinDoc RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Financial Report Intelligence</div>', unsafe_allow_html=True)

    # api_key = st.text_input("Google API Key", type="password", placeholder="AIzaSyBh...")
    # if api_key:
    #     os.environ["OPENAI_API_KEY"] = api_key
    

    st.markdown("---")
    st.markdown("**📂 Upload Document**")

    uploaded_file = st.file_uploader(
        "Annual Report / SEC Filing (PDF)",
        type=["pdf"],
        help="Upload any financial PDF — annual reports, 10-K, earnings releases, etc.",
    )

    use_sample = st.checkbox("Use BlackRock 2023 Annual Report sample", value=False)

    if st.button("⚡ Build Knowledge Base", use_container_width=True):
        # if not api_key:
        #     st.error("Please enter your OpenAI API key first.")
        # elif not uploaded_file and not use_sample:
        if not uploaded_file and not use_sample:
            st.error("Upload a PDF or check the sample option.")
        else:
            with st.spinner("Parsing PDF → chunking → embedding..."):
                try:
                    rag = FinancialRAG()
                    if uploaded_file:
                        pdf_bytes = uploaded_file.read()
                        stats = rag.ingest_pdf_bytes(pdf_bytes, filename=uploaded_file.name)
                    else:
                        stats = rag.ingest_sample()
                    st.session_state.rag = rag
                    st.session_state.doc_stats = stats
                    st.session_state.chat_history = []
                    st.success("Knowledge base ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Stats
    if st.session_state.doc_stats:
        st.markdown("---")
        s = st.session_state.doc_stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-num">{s.get("pages", "—")}</div><div class="metric-label">Pages</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-num">{s.get("chunks", "—")}</div><div class="metric-label">Chunks</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card" style="margin-top:8px"><div class="metric-num">{s.get("tokens", "—")}</div><div class="metric-label">Tokens Embedded</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.rag:
        st.markdown('<span class="badge-ready">● Knowledge Base Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-empty">○ No Document Loaded</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **📁 Where to get data**
    - [BlackRock Investor Relations](https://ir.blackrock.com)
    - [SEC EDGAR 10-K Filings](https://www.sec.gov/cgi-bin/browse-edgar)
    - [Annual Reports (.com)](https://www.annualreports.com)
    """)

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.rag:
                st.session_state.rag.clear_memory()
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 📊 Ask Your Financial Document")
st.markdown("Upload an annual report or 10-K, then ask anything — risk factors, revenue, strategy, AI investments.")
st.markdown("---")

# Sample questions as chips
if st.session_state.rag:
    st.markdown("**💡 Try asking:**")
    questions = get_sample_questions()
    cols = st.columns(3)
    for i, q in enumerate(questions):
        with cols[i % 3]:
            if st.button(q, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    st.markdown("---")

# Chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">🧑‍💼 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 Source passages", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card"><strong>Page {src["page"]}</strong> · Score: {src["score"]:.2f}<br>{src["text"]}</div>',
                            unsafe_allow_html=True,
                        )

# Blank state
if not st.session_state.rag:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #3a4a6a;">
      <div style="font-size:3rem">📑</div>
      <div style="font-size:1.1rem; margin-top:12px; color:#4a5f8a;">Upload a financial PDF and build the knowledge base to start asking questions.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask about the document... e.g. What are the key risk factors?")

# Handle chip click
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

if prompt and st.session_state.rag:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.spinner("Searching knowledge base..."):
        try:
            result = st.session_state.rag.query(prompt)
            answer = result["answer"]
            sources = result["sources"]
        except Exception as e:
            answer = f"Error: {e}"
            sources = []

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.rerun()

elif prompt and not st.session_state.rag:
    st.warning("Please upload a document and build the knowledge base first (sidebar).")
