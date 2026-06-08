from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
import os
import ssl
import certifi
import wikipedia
import arxiv
from arxiv import Search as ArxivSearch
from bs4 import BeautifulSoup
import requests
from tavily import TavilyClient
from dotenv import load_dotenv
from rich import print

load_dotenv()

# ── SSL Fix ────────────────────────────────────────────────────────────────────
ssl._create_default_https_context = ssl.create_default_context(
    cafile=certifi.where()
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ── Tavily Web Search ──────────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information.
    Returns Titles, URLs and snippets."""
    try:
        results = tavily.search(query=query, max_results=5)
        out = []
        for r in results["results"]:
            out.append(
                f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
            )
        return "\n----\n".join(out)
    except Exception as e:
        return f"Web search failed: {str(e)}"


# ── Wikipedia Search ───────────────────────────────────────────────────────────
@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for background and factual information about a topic."""
    try:
        wikipedia.set_user_agent("Mozilla/5.0 (research-tool)")
        _wiki_wrapper = WikipediaAPIWrapper(
            wiki_client=wikipedia,
            top_k_results=1,
            doc_content_chars_max=1500
        )
        return WikipediaQueryRun(api_wrapper=_wiki_wrapper).run(query)
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


# ── Arxiv Search ───────────────────────────────────────────────────────────────
@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for academic and scientific papers on a topic.
    Returns titles, authors, summaries and URLs."""
    try:
        _arxiv_wrapper = ArxivAPIWrapper(
            arxiv_search=ArxivSearch,
            arxiv_exceptions=(arxiv.ArxivError,),
            top_k_results=3
        )
        return ArxivQueryRun(api_wrapper=_arxiv_wrapper).run(query)
    except Exception as e:
        return f"Arxiv search failed: {str(e)}"


# ── URL Scraper ────────────────────────────────────────────────────────────────
@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading.
    Returns up to 1200 characters of clean body text."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:2000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"