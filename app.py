"""
Streamlit Frontend
----------------------------------------
Run:  streamlit run app.py
Requires FastAPI backend running at http://localhost:8000
"""

import os
import streamlit as st
import requests
import time
from datetime import datetime

# ── Config ──────────────────────────────────────────────────
# Reads API_BASE_URL env var so Docker Compose can inject the right address.
# Falls back to localhost for local development.

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Claims AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* Page background */
.stApp {
    background-color: #F5F2ED;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1A1A2E;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #E8E4DC !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #C9A96E !important;
    font-family: 'DM Serif Display', serif !important;
    border-bottom: 1px solid #2E2E4A;
    padding-bottom: 6px;
}

/* Header bar */
.header-bar {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 24px rgba(26,26,46,0.18);
}
.header-icon {
    font-size: 2.8rem;
    line-height: 1;
}
.header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #F5F2ED;
    margin: 0;
    line-height: 1.1;
}
.header-subtitle {
    font-size: 0.85rem;
    color: #8A9BB0;
    margin: 4px 0 0 0;
    font-weight: 300;
    letter-spacing: 0.04em;
}

/* Chat messages */
.msg-wrapper {
    margin-bottom: 20px;
}
.msg-user {
    background: #1A1A2E;
    color: #F5F2ED;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 20px;
    margin-left: 20%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(26,26,46,0.15);
}
.msg-ai {
    background: #FFFFFF;
    color: #1A1A2E;
    border-radius: 4px 18px 18px 18px;
    padding: 16px 22px;
    margin-right: 20%;
    font-size: 0.95rem;
    line-height: 1.7;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-left: 3px solid #C9A96E;
}
.msg-meta {
    font-size: 0.72rem;
    color: #9A8F85;
    margin-top: 6px;
    display: flex;
    gap: 12px;
    align-items: center;
}
.badge-cached {
    background: #E8F5E9;
    color: #2E7D32;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.68rem;
}
.badge-live {
    background: #FFF3E0;
    color: #E65100;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.68rem;
}

/* Sources expander */
.source-card {
    background: #F9F7F4;
    border: 1px solid #E5E0D8;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.83rem;
}
.source-doc {
    font-weight: 600;
    color: #1A1A2E;
    font-size: 0.8rem;
}
.source-score {
    color: #C9A96E;
    font-weight: 600;
}
.source-text {
    color: #6B6560;
    margin-top: 6px;
    line-height: 1.5;
    font-style: italic;
}

/* Metrics cards */
.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
}
.metric-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #1A1A2E;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.7rem;
    color: #9A8F85;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-online { background: #4CAF50; box-shadow: 0 0 6px #4CAF50; }
.status-offline { background: #F44336; }

/* Input area */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 1.5px solid #DDD8D0 !important;
    padding: 12px 16px !important;
    font-family: 'DM Sans', sans-serif !important;
    background: #FFFFFF !important;
    font-size: 0.95rem !important;
    color: #888 !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1A1A2E !important;
    box-shadow: 0 0 0 2px rgba(26,26,46,0.1) !important;
}

.stTextInput > div > div > input::placeholder {
    color: #888 !important;
    opacity: 1 !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    background: #1A1A2E !important;
    color: #F5F2ED !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #C9A96E !important;
    color: #1A1A2E !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(201,169,110,0.35) !important;
}

/* Divider */
hr { border-color: #E5E0D8; }

/* Suggestion chips */
.chip {
    display: inline-block;
    background: #FFFFFF;
    border: 1.5px solid #DDD8D0;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #444;
    cursor: pointer;
    margin: 4px;
    transition: all 0.15s;
}

.stSpinner > div {
    border-top-color: #1A1A2E !important;
    color: #888 !important;
}
</style>
""", unsafe_allow_html=True)


# ── API Helpers ───────────────────────────────────────────────
@st.cache_data(ttl=5)
def check_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def ask_question(question: str, top_k: int = 3, use_cache: bool = True) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "top_k": top_k, "use_cache": use_cache},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else {"error": r.json().get("detail", "Unknown error")}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the backend running?"}
    except Exception as e:
        return {"error": str(e)}
    
@st.cache_data(ttl=10)
def get_cache_stats() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/cache/stats", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None
    
@st.cache_data(ttl=10)
def get_cost_stats() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/cost/stats", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def clear_cache() -> bool:
    try:
        r = requests.delete(f"{API_BASE}/cache", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Claims AI")
    st.markdown("---")

    # Connection status
    health = check_health()
    if health:
        st.markdown(
            f'<span class="status-dot status-online"></span>'
            f'**API Online** — RAG {"✓" if health.get("rag_available") else "⚠ demo mode"}',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-dot status-offline"></span>**API Offline**',
            unsafe_allow_html=True,
        )
        st.warning("Start backend:\n```\nuvicorn insurance_claims_ai.api:app --reload\n```")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Context chunks", 1, 6, 3, help="Number of policy sections to retrieve")
    use_cache = st.toggle("Enable caching", value=True, help="Cache responses to reduce API costs")

    st.markdown("---")

    # Live metrics
    st.markdown("### 📊 Session Stats")
    cache_stats = get_cache_stats()
    cost_stats = get_cost_stats()

    if cache_stats:
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{cache_stats.get('total_requests', 0)}</div>
                    <div class="metric-label">Requests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{cache_stats.get('hit_rate_percent', 0):.0f}%</div>
                    <div class="metric-label">Cache Hit</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if cost_stats:
        savings = cost_stats.get("savings_usd", 0)
        total_cost = cost_stats.get("total_cost_usd", 0)
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">${total_cost:.4f}</div>
                    <div class="metric-label">Cost</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${savings:.4f}</div>
                    <div class="metric-label">Saved</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Cache", use_container_width=True):
            if clear_cache():
                st.success("Cleared")
                st.rerun()
    with col2:
        
        
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.72rem;color:#4A4A6A;text-align:center;">'
        "Insurance Claims AI v1.0<br>Powered by Claude + RAG</p>",
        unsafe_allow_html=True,
    )


# ── Main Content ──────────────────────────────────────────────
st.markdown(
    """
    <div class="header-bar">
        <div class="header-icon">🛡️</div>
        <div>
            <p class="header-title">Insurance Claims Assistant</p>
            <p class="header-subtitle">Ask anything about your policy — coverage, claims, deductibles, exclusions</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Suggestion chips (only show when no messages yet)
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    suggestions = [
        "What is my collision deductible?",
        "How do I file a claim?",
        "Is uninsured motorist coverage included?",
        "What is excluded from my policy?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_question = s

    st.markdown("---")

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-wrapper"><div class="msg-user">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        data = msg["data"]
        cached_badge = (
            '<span class="badge-cached">⚡ Cached</span>'
            if data.get("cached")
            else '<span class="badge-live">🔴 Live</span>'
        )
        latency = data.get("response_time_ms", 0)

        st.markdown(
            f"""
            <div class="msg-wrapper">
                <div class="msg-ai">{data.get("answer", "")}</div>
                <div class="msg-meta">
                    {cached_badge}
                    <span>{latency}ms</span>
                    <span>{data.get("timestamp", "")[:16].replace("T", " ")}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sources
        sources = data.get("sources", [])
        if sources:
            with st.expander(f"📄 {len(sources)} source{'s' if len(sources) != 1 else ''} retrieved"):
                for src in sources:
                    score_pct = int(src.get("similarity", 0) * 100)
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <span class="source-doc">📄 {src.get("document", "Unknown")}</span>
                                <span class="source-score">{score_pct}% match</span>
                            </div>
                            <div class="source-text">{src.get("text", "")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ── Input ──────────────────────────────────────────────────────
with st.form("question_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "question",
            placeholder="Ask about your insurance policy...",
            label_visibility="collapsed",
        )
    with col_btn:
        send = st.form_submit_button("Ask →")

final_question = ""
if send and question.strip():
    final_question = question.strip()
elif st.session_state.get("pending_question"):
    final_question = st.session_state.pending_question
    st.session_state.pending_question = ""

if final_question:
    already_added = (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and st.session_state.messages[-1]["content"] == final_question
    )

    if not already_added:
        st.session_state.messages.append({"role": "user", "content": final_question})

        if not health:
            st.session_state.messages.append({
                "role": "assistant",
                "data": {"answer": "⚠️ Backend is offline.", "sources": [], "cached": False, "response_time_ms": 0, "timestamp": datetime.now().isoformat()},
            })
        else:
            with st.spinner("Searching policy documents..."):
                result = ask_question(final_question, top_k=top_k, use_cache=use_cache)

            if result and "error" not in result:
                st.session_state.messages.append({"role": "assistant", "data": result})
            else:
                err = result.get("error", "Unknown error") if result else "No response"
                st.session_state.messages.append({
                    "role": "assistant",
                    "data": {"answer": f"❌ Error: {err}", "sources": [], "cached": False, "response_time_ms": 0, "timestamp": datetime.now().isoformat()},
                })

        st.rerun()