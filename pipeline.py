from agents import (
    build_search_agent,
    build_reader_agent,
    build_verifier_agent,
    writer_chain,
    critic_chain,
)
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import TypedDict
import re
import json
import os
from rich import print

# ── State ─────────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    topic: str
    search_results: str
    scraped_content: str
    research_combined: str
    report: str
    feedback: str
    score: int
    improvement_notes: str
    verified_report: str


# ── JSON History Log ──────────────────────────────────────────────────────────
HISTORY_FILE = "research_history.json"

def _load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"sessions": []}


def _save_to_history(topic: str, score: str):
    history = _load_history()
    history["sessions"].append({
        "topic": topic,
        "score": score,
        "date": str(date.today())
    })
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _check_already_researched(topic: str) -> bool:
    history = _load_history()
    for session in history["sessions"]:
        if session["topic"].lower() == topic.lower():
            print(f"\n[yellow]⚠️  You already researched this topic on {session['date']} with score {session['score']}[/yellow]")
            confirm = input("Do you want to research it again? (y/n): ")
            return confirm.strip().lower() != "y"
    return False


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_urls(text: str) -> list[str]:
    """Extract all URLs from search results text."""
    return re.findall(r'https?://[^\s\n"\']+', text)


def _extract_score(feedback: str) -> int:
    """Extract numeric score from critic feedback."""
    match = re.search(r'Score:\s*(\d+)/10', feedback)
    return int(match.group(1)) if match else 0


def _extract_areas_to_improve(feedback: str) -> str:
    """Extract only the Areas to Improve section from critic feedback."""
    match = re.search(r'Areas to Improve:(.*?)(?:One line verdict:|$)', feedback, re.DOTALL)
    return match.group(1).strip() if match else feedback


def _extract_key_findings(report: str) -> str:
    """Extract Key Findings section from the report."""
    match = re.search(r'Key Findings(.*?)(?:## Conclusion|## Sources|$)', report, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else report


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_research_pipeline(topic: str) -> ResearchState:

    # check history before starting
    if _check_already_researched(topic):
        print("\n[cyan]Skipping research. Check research_history.json for past results.[/cyan]")
        return {}

    state: ResearchState = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "research_combined": "",
        "report": "",
        "feedback": "",
        "score": 0,
        "improvement_notes": "",
        "verified_report": "",
    }

    # ── Step 1: Search ────────────────────────────────────────────────────────
    print("\n" + " =" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                f"Find recent, reliable and detailed information about: {topic}. "
                f"CRITICAL: You MUST use your custom tools to gather data. "
                f"Do NOT use your internal knowledge or built-in web search features. "
                f"Use web_search for current news and events. "
                f"Use wikipedia_search for background context. "
                f"Use arxiv_search only if the topic is academic or scientific. "
                f"Your final response must clearly output the Title and URL values provided by the tools. "
                f"Execute the tools now for the topic: {topic}."
            )
        }]
    })
    state["search_results"] = search_result["messages"][-1].content
    print("\nSearch Results:\n", state["search_results"])

    # ── Step 2: Read (multi-URL, parallel) ───────────────────────────────────
    print("\n" + " =" * 50)
    print("step 2 - reader agent is scraping top 3 URLs ...")
    print("=" * 50)

    urls = _extract_urls(state["search_results"])[:3]

    if urls:
        def scrape_single(url: str) -> str:
            reader_agent = build_reader_agent()
            result = reader_agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": (
                        f"Use the scrape_url tool to extract content from this URL: {url}. "
                        f"Return the raw content as-is, labeled with the URL."
                    )
                }]
            })
            return f"--- Source: {url} ---\n{result['messages'][-1].content}\n"

        with ThreadPoolExecutor(max_workers=3) as executor:
            scraped_results = list(executor.map(scrape_single, urls))

        state["scraped_content"] = "\n".join(scraped_results)
    else:
        print("[yellow]No URLs found in search results — skipping scraping.[/yellow]")
        state["scraped_content"] = "No scraped content available."

    print("\nScraped Content:\n", state["scraped_content"])

    # ── Step 3: Write + Feedback Loop ────────────────────────────────────────
    state["research_combined"] = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    MAX_ITERATIONS = 3
    SCORE_THRESHOLD = 7

    for iteration in range(1, MAX_ITERATIONS + 1):

        print("\n" + " =" * 50)
        print(f"step 3 - writer drafting report (iteration {iteration}/{MAX_ITERATIONS}) ...")
        print("=" * 50)

        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": state["research_combined"],
            "improvement_notes": state["improvement_notes"]
        })
        print("\nReport (iteration {}):\n".format(iteration), state["report"])

        # ── Step 4: Critic ────────────────────────────────────────────────────
        print("\n" + " =" * 50)
        print(f"step 4 - critic reviewing report (iteration {iteration}/{MAX_ITERATIONS}) ...")
        print("=" * 50)

        state["feedback"] = critic_chain.invoke({"report": state["report"]})
        state["score"] = _extract_score(state["feedback"])
        print("\nCritic Feedback:\n", state["feedback"])
        print(f"\n[bold]Score: {state['score']}/10[/bold]")

        # ── Check if good enough ──────────────────────────────────────────────
        if state["score"] >= SCORE_THRESHOLD:
            print(f"\n[green]✅ Score {state['score']}/10 meets threshold — stopping iterations.[/green]")
            break

        if iteration < MAX_ITERATIONS:
            state["improvement_notes"] = _extract_areas_to_improve(state["feedback"])
            print(f"\n[yellow]⚠️  Score {state['score']}/10 below threshold — sending back for revision...[/yellow]")
        else:
            print(f"\n[red]Max iterations reached. Proceeding with best report.[/red]")

    # ── Step 5: Verify ────────────────────────────────────────────────────────
    print("\n" + " =" * 50)
    print("step 5 - verifier agent checking key findings ...")
    print("=" * 50)

    key_findings = _extract_key_findings(state["report"])
    verifier_agent = build_verifier_agent()
    verifier_result = verifier_agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                f"Verify the following Key Findings from a research report about '{topic}'. "
                f"For each specific claim, number, or statistic, use wikipedia_search to verify it. "
                f"Mark verified claims with ✅ and unverified ones with ⚠️. "
                f"Return the annotated Key Findings only.\n\n"
                f"Key Findings:\n{key_findings}"
            )
        }]
    })
    state["verified_report"] = state["report"].replace(
        key_findings,
        verifier_result["messages"][-1].content
    )
    print("\nVerified Report:\n", state["verified_report"])

    # ── Step 6: Save to JSON history ──────────────────────────────────────────
    _save_to_history(topic, f"{state['score']}/10")
    print(f"\n[green]✅ Research saved to {HISTORY_FILE}[/green]")

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)