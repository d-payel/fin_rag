# 📊 FinDoc RAG — Financial Report Intelligence

A production-style **Retrieval-Augmented Generation (RAG)** system that lets you ask natural-language questions over any financial PDF — annual reports, SEC 10-K filings, earnings releases.

Built with **LangChain · ChromaDB · OpenAI · Streamlit**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────▼───────────────┐
          │    Streamlit Frontend (UI)    │
          └───────────────┬───────────────┘
                          │
          ┌───────────────▼───────────────────────────────────┐
          │              RAG Pipeline (LangChain)             │
          │                                                   │
          │  ┌──────────┐    ┌──────────┐    ┌───────────┐  │
          │  │  PDF     │    │  Text    │    │ Embedding │  │
          │  │ Parser   │───▶│ Chunker  │───▶│  (OpenAI) │  │
          │  │(PyMuPDF) │    │  800tok  │    │  ada-003  │  │
          │  └──────────┘    └──────────┘    └─────┬─────┘  │
          │                                        │         │
          │  ┌─────────────────────────────────────▼──────┐  │
          │  │          ChromaDB Vector Store              │  │
          │  │    (MMR Retrieval — top 5 diverse chunks)  │  │
          │  └─────────────────────────────────────┬──────┘  │
          │                                        │         │
          │  ┌─────────────────────────────────────▼──────┐  │
          │  │  ConversationalRetrievalChain (GPT-4o-mini) │  │
          │  │  + WindowMemory (last 5 turns)             │  │
          │  └─────────────────────────────────────┬──────┘  │
          │                                        │         │
          └────────────────────────────────────────┼─────────┘
                                                   │
          ┌────────────────────────────────────────▼─────────┐
          │   Answer + Source Citations (page, score, text)  │
          └───────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-turn chat** | Conversation memory (last 5 turns via `ConversationBufferWindowMemory`) |
| **Source citations** | Every answer shows source page, relevance score, and passage snippet |
| **MMR Retrieval** | Max Marginal Relevance ensures diverse, non-redundant context chunks |
| **Finance-tuned prompt** | System prompt engineered for financial precision, no hallucinated figures |
| **Sample questions** | One-click chip UI for quick demo |
| **Live stats** | Pages, chunks, and tokens embedded shown after ingestion |

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/financial-rag
cd financial-rag
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

Enter your **OpenAI API key** in the sidebar, upload a PDF, click **Build Knowledge Base**, and start asking.

---

## 📁 Where to Get Financial PDFs

### Free Official Sources

| Source | What you get | URL |
|---|---|---|
| **SEC EDGAR** | All US public company 10-K, 10-Q filings | https://www.sec.gov/cgi-bin/browse-edgar |
| **BlackRock IR** | BlackRock annual reports & earnings | https://ir.blackrock.com |
| **Annual Reports (.com)** | Curated annual reports from 8,000+ companies | https://www.annualreports.com |
| **Macrotrends** | Financial data + linked filings | https://www.macrotrends.net |

### Recommended Starter Documents

- **BlackRock 2023 Annual Report** — 10-K from EDGAR (search: `BLK 10-K 2023`)
- **JPMorgan Chase 2023 Annual Report** — good complexity
- **Apple 10-K 2023** — clean formatting, great for testing

### How to Download from SEC EDGAR

1. Go to https://efts.sec.gov/LATEST/search-index?q=%22BlackRock%22&dateRange=custom&startdt=2024-01-01&enddt=2024-12-31&forms=10-K
2. Click the filing → **Documents** tab → download the `.htm` or `.pdf` file

---

## 🗂️ Project Structure

```
financial-rag/
├── app.py                  # Streamlit UI
├── src/
│   ├── rag_pipeline.py     # Core RAG: PDF parsing, embedding, retrieval, chain
│   └── utils.py            # Helper functions
├── requirements.txt
└── README.md
```

---

## 🔧 Key Technical Decisions

**Chunking**: 800-token chunks with 120-token overlap — balances context richness with retrieval precision for dense financial prose.

**MMR Retrieval**: Chosen over simple similarity search to reduce redundancy when the same figure appears many times (e.g., AUM mentioned 50+ times in an annual report).

**GPT-4o-mini**: Cost-efficient for demo; swap to `gpt-4o` for production-grade accuracy.

**ChromaDB in-memory**: No setup needed; swap to persistent ChromaDB or Pinecone for production.

---

## 📈 Potential Extensions

- [ ] **Hybrid search** — combine vector + BM25 keyword search (LangChain `EnsembleRetriever`)
- [ ] **Multi-document** — compare two company annual reports side by side
- [ ] **RAGAS evaluation** — automated faithfulness + answer relevancy scoring
- [ ] **Table extraction** — use `pdfplumber` to parse financial tables as structured data
- [ ] **Streaming responses** — LangChain streaming callbacks for faster UX

---

## 🛠️ Built With

- [LangChain](https://langchain.com) — RAG orchestration
- [ChromaDB](https://trychroma.com) — vector store
- [OpenAI](https://openai.com) — embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF parsing
- [Streamlit](https://streamlit.io) — UI
