"""Utility helpers for the Streamlit app."""


def format_sources(sources: list[dict]) -> str:
    lines = []
    for src in sources:
        lines.append(f"**Page {src['page']}** (relevance: {src['score']:.2f})\n> {src['text'][:200]}...")
    return "\n\n".join(lines)


def get_sample_questions() -> list[str]:
    return [
        "What are the top risk factors?",
        "Summarise revenue and profit for 2023",
        "What is the AUM and how did it change?",
        "What does the company say about AI investments?",
        "What is the dividend and buyback policy?",
        "What acquisitions were made recently?",
    ]
