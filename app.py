import streamlit as st
import time
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from agents import (
    build_search_agent,
    build_reader_agent,
    build_verifier_agent,
    writer_chain,
    critic_chain,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #080b12;
    --bg2:       #0d1117;
    --bg3:       #111720;
    --border:    rgba(255,255,255,0.07);
    --border-hi: rgba(99,179,237,0.35);
    --accent:    #63b3ed;
    --accent2:   #4299e1;
    --green:     #68d391;
    --amber:     #f6ad55;
    --red:       #fc8181;
    --text:      #e2e8f0;
    --text-dim:  #a0aec0;
    --text-faint:#2d3748;
    --font:      'Outfit', sans-serif;
    --mono:      'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font);
    color: var(--text);
    background: var(--bg);
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 70% 50% at 15% 0%, rgba(99,179,237,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 85% 100%, rgba(66,153,225,0.05) 0%, transparent 55%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 4rem; max-width: 1300px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* ── Hide ghost empty block above input ── */
.block-container > div > div:first-child > div[data-testid="stVerticalBlock"] > div:first-child > div:empty {
    display: none !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
    outline: none !important;
}
.stTextInput > label {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
    font-weight: 400 !important;
    margin-bottom: 0.4rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.12em !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.8rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(99,179,237,0.08) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px rgba(99,179,237,0.15) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(104,211,145,0.08) !important;
    color: var(--green) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    border: 1px solid rgba(104,211,145,0.25) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: rgba(104,211,145,0.15) !important;
    box-shadow: 0 0 16px rgba(104,211,145,0.1) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.1em !important;
}
.streamlit-expanderContent {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Report content styling ── */
.report-body h1, .report-body h2, .report-body h3 {
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.6rem !important;
}
.report-body h1 { font-size: 1.5rem !important; }
.report-body h2 { font-size: 1.2rem !important; color: #63b3ed !important; }
.report-body h3 { font-size: 1rem !important; }
.report-body p, .report-body li {
    color: #f0f4f8 !important;
    font-size: 0.94rem !important;
    line-height: 1.85 !important;
    opacity: 1 !important;
}
.report-body strong { color: #ffffff !important; font-weight: 600 !important; }
.report-body a { color: var(--accent) !important; }
/* Override Streamlit default dim text inside panels */
.report-body [data-testid="stMarkdownContainer"] p,
.report-body [data-testid="stMarkdownContainer"] li {
    color: #f0f4f8 !important;
    opacity: 1 !important;
}
/* Global override for all text inside result panels */
div[style*="bg2"] p, div[style*="bg2"] li,
div[style*="report-body"] p, div[style*="report-body"] li {
    color: #f0f4f8 !important;
}

/* ── Spinner ── */
.stSpinner > div { color: var(--accent) !important; }

/* ── Force bright text in all markdown containers ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th {
    color: #f0f4f8 !important;
    opacity: 1 !important;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #ffffff !important;
    font-family: var(--font) !important;
    font-weight: 700 !important;
}
[data-testid="stMarkdownContainer"] h2 {
    color: var(--accent) !important;
}
[data-testid="stMarkdownContainer"] strong {
    color: #ffffff !important;
}
[data-testid="stMarkdownContainer"] a {
    color: var(--accent) !important;
}

/* ── Warning ── */
.stWarning {
    background: rgba(246,173,85,0.08) !important;
    border: 1px solid rgba(246,173,85,0.25) !important;
    border-radius: 8px !important;
    color: var(--amber) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
HISTORY_FILE = "research_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"sessions": []}

def save_to_history(topic, score):
    history = load_history()
    history["sessions"].append({
        "topic": topic,
        "score": score,
        "date": str(date.today())
    })
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def extract_urls(text):
    return re.findall(r'https?://[^\s\n"\']+', text)

def extract_score(feedback):
    match = re.search(r'Score:\s*(\d+(?:\.\d+)?)/10', feedback)
    return float(match.group(1)) if match else 0.0

def extract_areas_to_improve(feedback):
    match = re.search(r'Areas to Improve:(.*?)(?:One line verdict:|$)', feedback, re.DOTALL)
    return match.group(1).strip() if match else feedback

def extract_key_findings(report):
    match = re.search(r'(#{1,3}\s*\*{0,2}Key Findings\*{0,2}.*?)(?=#{1,3}\s*(Conclusion|Sources)|$)',
                      report, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def clean_report(text):
    """Clean malformed markdown artifacts from LLM output."""
    # Fix merged headings: **Key Findings### Annotated Key Findings -> ## Annotated Key Findings
    text = re.sub(r"[*]{1,2}[^*\n]*[#]{2,3}\s*([^\n]+)", r"## \1", text)
    # Fix **heading** on its own line -> ### heading
    text = re.sub(r"^[*]{2}([^*\n]{3,80})[*]{2}\s*$", r"### \1", text, flags=re.MULTILINE)
    # Fix __heading__ on its own line -> ### heading
    text = re.sub(r"^[_]{2}([^_\n]{3,80})[_]{2}\s*$", r"### \1", text, flags=re.MULTILINE)
    # Fix ###heading missing space -> ### heading
    text = re.sub(r"^([#]{1,4})([^#\s])", r"\1 \2", text, flags=re.MULTILINE)
    return text

def score_color(score):
    if score >= 8: return "var(--green)"
    if score >= 6: return "var(--amber)"
    return "var(--red)"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-family:var(--mono);font-size:0.65rem;letter-spacing:0.2em;
                    text-transform:uppercase;color:var(--accent);margin-bottom:0.3rem;">
            ⬡ ResearchMind
        </div>
        <div style="font-size:0.8rem;color:var(--text-dim);line-height:1.5;">
            Multi-agent AI research pipeline
        </div>
    </div>
    <div style="height:1px;background:var(--border);margin-bottom:1.5rem;"></div>
    <div style="font-family:var(--mono);font-size:0.65rem;letter-spacing:0.18em;
                text-transform:uppercase;color:var(--text-dim);margin-bottom:1rem;">
        Research History
    </div>
    """, unsafe_allow_html=True)

    history = load_history()
    sessions = history.get("sessions", [])

    if not sessions:
        st.markdown("""
        <div style="font-size:0.78rem;color:var(--text-faint);font-family:var(--mono);
                    padding:0.8rem;border:1px dashed rgba(255,255,255,0.05);
                    border-radius:8px;text-align:center;">
            No research yet
        </div>
        """, unsafe_allow_html=True)
    else:
        for s in reversed(sessions[-10:]):
            score_val = float(s['score'].split('/')[0]) if '/' in s['score'] else 0
            col = score_color(score_val)
            st.markdown(f"""
            <div style="background:var(--bg3);border:1px solid var(--border);
                        border-radius:8px;padding:0.7rem 0.9rem;margin-bottom:0.5rem;">
                <div style="font-size:0.82rem;color:var(--text);font-weight:500;
                            margin-bottom:0.3rem;white-space:nowrap;overflow:hidden;
                            text-overflow:ellipsis;">
                    {s['topic'][:35]}{'...' if len(s['topic']) > 35 else ''}
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:var(--mono);font-size:0.65rem;
                                 color:var(--text-dim);">{s['date']}</span>
                    <span style="font-family:var(--mono);font-size:0.72rem;
                                 font-weight:600;color:{col};">{s['score']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="height:1px;background:var(--border);margin:1.5rem 0;"></div>
    <div style="font-family:var(--mono);font-size:0.62rem;color:var(--text-faint);line-height:1.7;">
        <div>Search → Read → Write</div>
        <div>Critique → Refine → Verify</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:3rem 0 2rem;">
    <div style="font-family:var(--mono);font-size:0.65rem;letter-spacing:0.25em;
                text-transform:uppercase;color:var(--accent);opacity:0.8;margin-bottom:0.8rem;">
        Multi-Agent Research System
    </div>
    <h1 style="font-family:var(--font);font-size:clamp(2.5rem,5vw,4rem);
               font-weight:800;line-height:1.05;letter-spacing:-0.03em;
               color:var(--text);margin:0 0 0.8rem;">
        Research<span style="color:var(--accent);">Mind</span>
    </h1>
    <p style="font-size:1rem;font-weight:300;color:var(--text-dim);
              max-width:480px;line-height:1.7;margin:0;">
        Five specialized agents — searching, scraping, writing,
        critiquing, and verifying — deliver a polished, fact-checked report.
    </p>
</div>
<div style="height:1px;background:linear-gradient(90deg,transparent,
            rgba(99,179,237,0.25),transparent);margin-bottom:2.5rem;"></div>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("results", {}), ("running", False), ("done", False),
             ("current_step", ""), ("iteration", 0), ("final_score", 0.0)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_gap, col_right = st.columns([5, 0.4, 3.5])

with col_left:
    # Input — no wrapper div that creates ghost box
    topic = st.text_input(
        "RESEARCH TOPIC",
        placeholder="e.g. How CRISPR is changing medicine in 2025",
        key="topic_input",
    )
    run_btn = st.button("⬡  EXECUTE PIPELINE", use_container_width=True)

    st.markdown("""
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;margin-top:1rem;margin-bottom:2rem;">
        <span style="font-family:var(--mono);font-size:0.62rem;color:var(--text-faint);letter-spacing:0.1em;">EXAMPLES →</span>
        <span style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;
                     padding:0.2rem 0.65rem;font-size:0.72rem;color:var(--text-dim);">LLM agents 2025</span>
        <span style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;
                     padding:0.2rem 0.65rem;font-size:0.72rem;color:var(--text-dim);">Fusion energy progress</span>
        <span style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;
                     padding:0.2rem 0.65rem;font-size:0.72rem;color:var(--text-dim);">Quantum computing</span>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div style="font-family:var(--mono);font-size:0.65rem;letter-spacing:0.18em;
                text-transform:uppercase;color:var(--text-dim);margin-bottom:1rem;margin-top:0.2rem;">
        Pipeline Steps
    </div>
    """, unsafe_allow_html=True)

    r = st.session_state.results
    cur = st.session_state.current_step
    itr = st.session_state.iteration

    steps = [
        ("01", "Search Agent",   "search",   "Web + Wikipedia + Arxiv"),
        ("02", "Reader Agent",   "reader",   "Scrapes top 3 URLs"),
        ("03", "Write & Refine", "writer",   f"Iteration {itr}/3" if itr > 0 else "Feedback loop"),
        ("04", "Critic",         "critic",   "Scores the report"),
        ("05", "Verifier",       "verifier", "Fact-checks key findings"),
    ]

    for num, title, key, desc in steps:
        if key in r:
            left_col = "var(--green)"
            card_bg = "rgba(104,211,145,0.03)"
            card_border = "rgba(104,211,145,0.2)"
            status_txt = "✓ DONE"
            status_col = "var(--green)"
        elif cur == key:
            left_col = "var(--amber)"
            card_bg = "rgba(246,173,85,0.04)"
            card_border = "rgba(246,173,85,0.3)"
            status_txt = "● RUNNING"
            status_col = "var(--amber)"
        else:
            left_col = "rgba(255,255,255,0.05)"
            card_bg = "transparent"
            card_border = "var(--border)"
            status_txt = "WAITING"
            status_col = "var(--text-faint)"

        st.markdown(f"""
        <div style="background:{card_bg};border:1px solid {card_border};
                    border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:0.6rem;
                    border-left:3px solid {left_col};">
            <div style="display:flex;align-items:center;justify-content:space-between;">
                <div style="display:flex;align-items:center;gap:0.6rem;">
                    <span style="font-family:var(--mono);font-size:0.6rem;color:var(--accent);opacity:0.6;">{num}</span>
                    <span style="font-family:var(--font);font-size:0.88rem;font-weight:600;color:var(--text);">{title}</span>
                </div>
                <span style="font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;color:{status_col};">{status_txt}</span>
            </div>
            <div style="font-size:0.72rem;color:var(--text-dim);margin-top:0.25rem;padding-left:1.4rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.final_score > 0:
        sc = st.session_state.final_score
        col = score_color(sc)
        st.markdown(f"""
        <div style="background:var(--bg3);border:1px solid {col}33;border-radius:10px;
                    padding:0.9rem 1.1rem;margin-top:0.5rem;text-align:center;">
            <div style="font-family:var(--mono);font-size:0.62rem;color:var(--text-dim);
                        letter-spacing:0.15em;margin-bottom:0.3rem;">FINAL SCORE</div>
            <div style="font-family:var(--font);font-size:2rem;font-weight:800;color:{col};">
                {sc:g}<span style="font-size:1rem;font-weight:400;color:var(--text-dim);">/10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Trigger ────────────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.current_step = ""
        st.session_state.iteration = 0
        st.session_state.final_score = 0.0
        st.rerun()


# ── Pipeline execution ─────────────────────────────────────────────────────────
if st.session_state.running and not st.session_state.done:
    results = {}
    topic_val = st.session_state.topic_input

    # Step 1: Search
    st.session_state.current_step = "search"
    with st.spinner("🔍  Search Agent scanning the web…"):
        search_agent = build_search_agent()
        sr = search_agent.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Find recent, reliable and detailed information about: {topic_val}. "
                    f"CRITICAL: You MUST use your custom tools only. "
                    f"Do NOT use internal knowledge or built-in search. "
                    f"Use web_search for news, wikipedia_search for background, "
                    f"arxiv_search only if academic. Return all Titles and URLs clearly."
                )
            }]
        })
        results["search"] = sr["messages"][-1].content
        st.session_state.results = dict(results)

    # Step 2: Reader
    st.session_state.current_step = "reader"
    with st.spinner("📄  Reader Agent scraping top 3 sources…"):
        urls = extract_urls(results["search"])[:3]
        if urls:
            def scrape_single(url):
                reader_agent = build_reader_agent()
                res = reader_agent.invoke({
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"Use scrape_url tool to extract content from: {url}. "
                            f"Return raw content labeled with the URL."
                        )
                    }]
                })
                return f"--- Source: {url} ---\n{res['messages'][-1].content}\n"

            with ThreadPoolExecutor(max_workers=3) as ex:
                scraped = list(ex.map(scrape_single, urls))
            results["reader"] = "\n".join(scraped)
        else:
            results["reader"] = "No URLs found in search results."
        st.session_state.results = dict(results)

    # Step 3 + 4: Write + Critic loop
    research_combined = (
        f"SEARCH RESULTS:\n{results['search']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
    )

    MAX_ITER = 3
    THRESHOLD = 7
    improvement_notes = ""
    final_report = ""
    final_feedback = ""
    final_score = 0

    for iteration in range(1, MAX_ITER + 1):
        st.session_state.iteration = iteration
        st.session_state.current_step = "writer"

        with st.spinner(f"✍️  Writing report — iteration {iteration}/{MAX_ITER}…"):
            final_report = writer_chain.invoke({
                "topic": topic_val,
                "research": research_combined,
                "improvement_notes": improvement_notes
            })

        st.session_state.current_step = "critic"
        with st.spinner(f"🧐  Critic reviewing — iteration {iteration}/{MAX_ITER}…"):
            final_feedback = critic_chain.invoke({"report": final_report})
            final_score = extract_score(final_feedback)
            st.session_state.final_score = final_score

        if final_score >= THRESHOLD:
            break
        if iteration < MAX_ITER:
            improvement_notes = extract_areas_to_improve(final_feedback)

    results["writer"] = final_report
    results["critic"] = final_feedback
    st.session_state.results = dict(results)

    # Step 5: Verifier
    st.session_state.current_step = "verifier"
    with st.spinner("🔬  Verifier fact-checking key findings…"):
        key_findings = extract_key_findings(final_report)
        if key_findings:
            verifier_agent = build_verifier_agent()
            vr = verifier_agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": (
                        f"Verify the Key Findings from a report about '{topic_val}'. "
                        f"For each claim, use wikipedia_search to verify. "
                        f"Mark verified with ✅, unverified with ⚠️. "
                        f"Return annotated Key Findings only.\n\n{key_findings}"
                    )
                }]
            })
            verified_findings = vr["messages"][-1].content
            verified_report = final_report.replace(key_findings, verified_findings)
        else:
            verified_report = final_report

        results["verifier"] = clean_report(verified_report)
        st.session_state.results = dict(results)

    save_to_history(topic_val, f"{final_score}/10")
    st.session_state.running = False
    st.session_state.done = True
    st.session_state.current_step = ""
    st.rerun()


# ── Results ────────────────────────────────────────────────────────────────────
r = st.session_state.results

if r and st.session_state.done:
    st.markdown("""
    <div style="height:1px;background:linear-gradient(90deg,transparent,
                rgba(99,179,237,0.2),transparent);margin:2rem 0;"></div>
    """, unsafe_allow_html=True)

    # Raw outputs
    col_a, col_b = st.columns(2)
    with col_a:
        if "search" in r:
            with st.expander("🔍 Search Results (raw)"):
                st.markdown(f"""
                <div style="font-family:var(--mono);font-size:0.75rem;color:var(--text-dim);
                            line-height:1.7;white-space:pre-wrap;">{r['search']}</div>
                """, unsafe_allow_html=True)
    with col_b:
        if "reader" in r:
            with st.expander("📄 Scraped Content (raw)"):
                st.code(r["reader"][:3000], language=None)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Final verified report
    if "verifier" in r:
        st.markdown("""
        <div style="background:var(--bg2);border:1px solid rgba(99,179,237,0.2);
                    border-radius:14px;padding:2rem 2.5rem;margin-bottom:1.5rem;">
            <div style="font-family:var(--font);font-size:1.4rem;font-weight:800;
                        color:var(--accent);margin-bottom:1.2rem;padding-bottom:0.8rem;
                        border-bottom:1px solid rgba(99,179,237,0.15);
                        letter-spacing:-0.01em;">
                ⬡ Final Research Report — Verified
            </div>
            <div class="report-body">
        """, unsafe_allow_html=True)
        st.markdown(r["verifier"])
        st.markdown("</div></div>", unsafe_allow_html=True)

        col_dl, col_sp = st.columns([2, 5])
        with col_dl:
            st.download_button(
                label="⬇  Download Report (.md)",
                data=r["verifier"],
                file_name=f"research_{int(time.time())}.md",
                mime="text/markdown",
            )

    # Critic feedback
    if "critic" in r:
        sc = st.session_state.final_score
        col = score_color(sc)
        st.markdown(f"""
        <div style="background:var(--bg2);border:1px solid rgba(104,211,145,0.2);
                    border-radius:14px;padding:2rem 2.5rem;margin-top:1.5rem;">
            <div style="display:flex;align-items:center;justify-content:space-between;
                        margin-bottom:1.2rem;padding-bottom:0.8rem;
                        border-bottom:1px solid rgba(104,211,145,0.1);">
                <span style="font-family:var(--font);font-size:1.4rem;font-weight:800;
                             color:var(--green);letter-spacing:-0.01em;">
                    ⬡ Critic Feedback
                </span>
                <span style="font-family:var(--font);font-size:1.6rem;
                             font-weight:800;color:{col};">
                    {sc:g}<span style="font-size:1rem;font-weight:400;color:var(--text-dim);">/10</span>
                </span>
            </div>
            <div class="report-body">
        """, unsafe_allow_html=True)
        st.markdown(r["critic"])
        st.markdown("</div></div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:var(--mono);font-size:0.6rem;color:var(--text-faint);
            text-align:center;margin-top:4rem;letter-spacing:0.1em;">
    ResearchMind · LangChain Multi-Agent Pipeline · Streamlit
</div>
""", unsafe_allow_html=True)
