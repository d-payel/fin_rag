"""
FinancialRAG — Core RAG Pipeline
Uses: LangChain + ChromaDB + LLama
"""
import os
import re
from typing import Any

import fitz
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


FINANCE_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "input"],
    template="""You are a senior financial analyst assistant. You answer questions about financial documents
(annual reports, 10-K filings, earnings releases) with precision and clarity.

Guidelines:
- Be specific and cite figures, percentages, and dates when available in the context.
- If the answer is not in the context, say "This information is not available in the uploaded document."
- Keep answers concise but complete. Use bullet points for multi-part answers.
- Never hallucinate financial figures.

Retrieved context from the document:
{context}

Conversation history:
{chat_history}

Question: {input}

Answer:"""
)


class FinancialRAG:
    # def __init__(self, api_key: str, groq_api_key: str, model: str):
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.groq_api_key = os.environ.get("GROQ_API_KEY")

        # Gemini embeddings — free tier
        
        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/gemini-embedding-001",   # ← change to this
        #     google_api_key=self.api_key,
        # )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # self.llm = ChatGoogleGenerativeAI(
            
        #     model="models/gemini-2.0-flash-lite",
        #     temperature=0.1,
        #     google_api_key=api_key,
        # )
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            api_key=self.groq_api_key,
        )

        self.vectorstore = None
        self.chain = None

        # Simple manual memory (list of dicts) — no langchain_classic needed
        self.chat_history: list[dict] = []

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def _parse_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> list[Document]:
        """Convert PDF bytes → list of LangChain Documents (one per page)."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text = self._clean_text(text)
            if len(text.strip()) < 50:
                continue
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num + 1,
                    "total_pages": len(doc),
                }
            ))
        doc.close()
        return documents

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        return text.strip()

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split page-level docs into smaller overlapping chunks."""
        chunks = []
        for doc in documents:
            splits = self.splitter.split_text(doc.page_content)
            for i, split in enumerate(splits):
                chunks.append(Document(
                    page_content=split,
                    metadata={**doc.metadata, "chunk": i}
                ))
        return chunks

    def _build_vectorstore(self, chunks: list[Document]) -> Chroma:
        """Embed chunks and store in ChromaDB (in-memory)."""
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="financial_docs",
            collection_metadata={"hnsw:space": "cosine"},
        )
        return vectorstore

    # def _build_chain(self):
    #     """Build the retrieval chain."""
    #     retriever = self.vectorstore.as_retriever(
    #         search_type="mmr",
    #         search_kwargs={"k": 5, "fetch_k": 20},
    #     )
    #     combine_docs_chain = create_stuff_documents_chain(self.llm, FINANCE_PROMPT)
    #     self.chain = create_retrieval_chain(retriever, combine_docs_chain)
    def _build_chain(self):
        """Build LCEL retrieval chain — modern standard, no deprecated classes."""
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )

        def format_docs(docs):
            self._last_docs = docs  # store for source citation
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "input": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: self._get_history()),
            }
            | FINANCE_PROMPT
            | self.llm
            | StrOutputParser()
        )
        self._retriever = retriever
        self._last_docs = []
    def _get_history(self) -> str:
        history_text = ""
        for turn in self.chat_history[-5:]:
            history_text += f"Human: {turn['human']}\nAssistant: {turn['assistant']}\n"
        return history_text


    def ingest_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> dict:
        """Full ingestion pipeline: PDF bytes → vectorstore → chain."""
        pages = self._parse_pdf_bytes(pdf_bytes, filename)
        chunks = self._chunk_documents(pages)
        total_chars = sum(len(c.page_content) for c in chunks)
        self.vectorstore = self._build_vectorstore(chunks)
        self._build_chain()
        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "tokens": f"~{total_chars // 4:,}",
        }

    def ingest_sample(self) -> dict:
        """Ingest a built-in sample text (for demo without a PDF)."""
        sample_text = """
BlackRock, Inc. Annual Report 2023

BUSINESS OVERVIEW
BlackRock is the world's largest asset manager with $10 trillion in assets under management (AUM) as of December 31, 2023.
The firm provides investment management, risk management, and advisory services to institutional and retail clients globally.
BlackRock operates through three primary segments: Investment Management, Technology Services (Aladdin), and Advisory Services.

FINANCIAL HIGHLIGHTS 2023
- Total revenue: $17.9 billion (up 7% year-over-year)
- Operating income: $6.6 billion
- Net income attributable to BlackRock: $5.5 billion
- Diluted EPS: $36.51
- Assets under management: $10.0 trillion (record high)
- Long-term net inflows: $289 billion

ALADDIN TECHNOLOGY PLATFORM
BlackRock's Aladdin platform generated $1.5 billion in technology services revenue in 2023, representing 12% growth.
Aladdin serves over 1,000 clients including insurance companies, pension funds, and sovereign wealth funds.
The platform processes risk analytics for over $21 trillion in assets globally.
BlackRock continued to invest heavily in AI and machine learning capabilities within Aladdin.
New AI-powered features include portfolio construction tools, ESG analytics, and real-time risk monitoring.

RISK FACTORS
Market Risk: Adverse changes in financial markets could significantly reduce AUM and therefore management fees.
Regulatory Risk: As a global firm, BlackRock is subject to regulations in over 30 jurisdictions; changes could increase compliance costs.
Operational Risk: Reliance on technology systems including Aladdin creates exposure to cyber threats and system outages.
Competitive Risk: Increasing competition from passive investment providers and fintech firms pressures fees.
Key Person Risk: The company's success depends significantly on senior leadership including CEO Larry Fink.
Climate Risk: Physical and transition risks related to climate change could impact investment portfolios.

STRATEGY AND OUTLOOK
BlackRock's "Whole Portfolio" strategy aims to provide comprehensive investment solutions across public and private markets.
The firm is expanding its presence in private markets through partnerships and acquisitions.
In 2023, BlackRock acquired Kreos Capital and SpiderRock Advisors to expand alternative investment capabilities.
The company announced the pending acquisition of Global Infrastructure Partners (GIP) for $12.5 billion in January 2024.
BlackRock plans to integrate AI across all business lines to improve investment outcomes and operational efficiency.
The firm launched several new AI-powered ETFs and thematic investment products in 2023.

ESG AND SUSTAINABILITY
BlackRock manages over $700 billion in sustainable investment strategies.
The firm signed the Net Zero Asset Managers Initiative and has committed to supporting portfolio companies' transition to net zero.
BlackRock published its 2023 Sustainability Report with detailed portfolio carbon metrics.

SHAREHOLDER RETURNS
BlackRock returned $4.6 billion to shareholders in 2023 through dividends and share buybacks.
The annual dividend was increased by 2% to $20.40 per share.
Share repurchases totaled $1.8 billion during the year.
        """
        chunks = self.splitter.split_text(sample_text)
        chunk_docs = [
            Document(page_content=c, metadata={"source": "BlackRock_Sample_2023.pdf", "page": 1, "chunk": i})
            for i, c in enumerate(chunks)
        ]
        self.vectorstore = self._build_vectorstore(chunk_docs)
        self._build_chain()
        return {"pages": 1, "chunks": len(chunk_docs), "tokens": f"~{len(sample_text) // 4:,}"}

    # ── Querying ──────────────────────────────────────────────────────────────

    # def query(self, question: str) -> dict[str, Any]:
    #     """Run a query through the RAG chain and return answer + sources."""
    #     if not self.chain:
    #         raise RuntimeError("No document ingested. Call ingest_pdf_bytes() first.")

    #     # Format last 5 exchanges as a plain string for the prompt
    #     history_text = ""
    #     for turn in self.chat_history[-5:]:
    #         history_text += f"Human: {turn['human']}\nAssistant: {turn['assistant']}\n"

    #     result = self.chain.invoke({
    #         "input": question,
    #         "chat_history": history_text,
    #     })

    #     answer = result["answer"]

    #     # Save to memory
    #     self.chat_history.append({"human": question, "assistant": answer})

    #     sources = []
    #     for doc in result.get("context", []):
    #         sources.append({
    #             "page": doc.metadata.get("page", "?"),
    #             "text": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
    #             "score": self._get_similarity_score(question, doc.page_content),
    #         })

    #     return {"answer": answer, "sources": sources}

    # def query(self, question: str) -> dict[str, Any]:
    #     if not self.chain:
    #         raise RuntimeError("No document ingested.")

    #     self._last_docs = []
    #     answer = self.chain.invoke(question)

    #     self.chat_history.append({"human": question, "assistant": answer})

    #     sources = []
    #     for doc in self._last_docs:
    #         sources.append({
    #             "page": doc.metadata.get("page", "?"),
    #             "text": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
    #             "score": self._get_similarity_score(question, doc.page_content),
    #         })

    #     return {"answer": answer, "sources": sources}
    def query(self, question: str) -> dict[str, Any]:

        if not self.chain:
            raise RuntimeError("No document ingested.")
        # Get docs with scores directly from vectorstore
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            question, k=5
        )
        
        # Then pass docs to chain
        
        answer = self.chain.invoke(question)

        self.chat_history.append({"human": question, "assistant": answer})

        
        sources = []
        for doc, score in docs_with_scores:
            sources.append({
                "page": doc.metadata.get("page", "?"),
                "text": doc.page_content[:300] + "...",
                "score": round(float(score), 3),
            })
        
        self.chat_history.append({"human": question, "assistant": answer})
        return {"answer": answer, "sources": sources}

    def _get_similarity_score(self, query: str, text: str) -> float:
        """Approximate similarity score using embedding dot product."""
        try:
            q_emb = self.embeddings.embed_query(query)
            t_emb = self.embeddings.embed_query(text[:500])
            dot = sum(a * b for a, b in zip(q_emb, t_emb))
            return round(min(max(dot, 0.0), 1.0), 3)
        except Exception:
            return 0.0

    def clear_memory(self):
        """Reset conversation memory."""
        self.chat_history = []
