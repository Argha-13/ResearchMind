# ⬡ ResearchMind — Multi-Agent AI Research Pipeline

A multi-agent AI system that searches, scrapes, writes, critiques, refines, and fact-checks research reports autonomously — all from a single topic input.

---

## What It Does

ResearchMind runs a 5-step agentic pipeline on any research topic:

1. **Search Agent** — queries the web (Tavily), Wikipedia, and Arxiv to gather sources
2. **Reader Agent** — scrapes the top 3 URLs in parallel for deeper content
3. **Writer Chain** — drafts a structured research report (Introduction, Key Findings, Conclusion, Sources)
4. **Critic Chain** — scores the report out of 10 and identifies specific areas to improve
5. **Feedback Loop** — if the score is below 7, the writer revises the report using only the critic's improvement notes (max 3 iterations)
6. **Verifier Agent** — fact-checks each claim in the Key Findings section against Wikipedia, annotating with ✅ or ⚠️

All results are displayed in a clean Streamlit UI with a persistent research history log.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Mistral Medium 3 via `langchain-mistralai` |
| Agents | LangChain `create_agent` |
| Web Search | Tavily API |
| Encyclopedic Search | Wikipedia API |
| Academic Search | Arxiv API |
| Web Scraping | BeautifulSoup + Requests |
| Frontend | Streamlit |
| Parallelism | Python `ThreadPoolExecutor` |
| History | JSON file persistence |

---

## Project Structure

```
multiagent_system/
├── app.py                  # Streamlit UI
├── pipeline.py             # Core pipeline logic (CLI version)
├── agents.py               # Agent and chain definitions
├── tools.py                # Tool definitions (search, scrape, wiki, arxiv)
├── main.py                 # Entry point for CLI
├── research_history.json   # Auto-generated research history log
├── requirements.txt        # Dependencies
├── .env                    # API keys (never commit this)
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/multiagent_system.git
cd multiagent_system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```
TAVILY_API_KEY=your_tavily_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get your API keys:
- Tavily: https://app.tavily.com
- Mistral: https://console.mistral.ai

### 5. Run the app

```bash
# Streamlit UI
streamlit run app.py

# CLI version
python pipeline.py
```

---

## Pipeline Architecture

```
Topic Input
    │
    ▼
┌─────────────────┐
│  Search Agent   │  ← web_search + wikipedia_search + arxiv_search
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reader Agent   │  ← scrape_url × 3 (parallel)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Writer + Critic Loop        │
│                                 │
│  Writer Chain → Critic Chain    │
│       ↑              │          │
│       └── if score < 7/10 ──────┘  max 3 iterations
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Verifier Agent  │  ← wikipedia_search (fact-check Key Findings)
└────────┬────────┘
         │
         ▼
  Final Verified Report
  + JSON History Log
```

---

## Key Features

- **Iterative refinement** — critic feedback directly improves the report, not just scores it
- **Parallel scraping** — top 3 URLs scraped simultaneously via `ThreadPoolExecutor`
- **Fact verification** — Key Findings annotated with ✅ verified / ⚠️ unverified
- **Persistent history** — every research session logged with topic, score, and date
- **Graceful error handling** — all tools wrapped in try/except, pipeline never crashes on a single tool failure
- **Topic deduplication** — warns if you've already researched the same topic

---

## Environment Variables

| Variable | Description |
|---|---|
| `TAVILY_API_KEY` | Required — Tavily web search API key |
| `MISTRAL_API_KEY` | Required — Mistral LLM API key |

Wikipedia and Arxiv require no API keys.

---

## Requirements

```
langchain
langchain-mistralai
langchain-community
tavily-python
beautifulsoup4
requests
wikipedia
arxiv
streamlit
python-dotenv
rich
certifi
```

---

## Roadmap

- [ ] Conversational interface — ask follow-up questions about past reports
- [ ] Vector store (FAISS/Chroma) — semantic search over research history
- [ ] FastAPI backend + React frontend
- [ ] Export to PDF
- [ ] Multi-language support

---

## License

MIT License — feel free to use, modify, and build on this project.

---

Built with LangChain · Mistral · Streamlit
