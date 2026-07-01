import streamlit as st
import streamlit.components.v1 as components
import time
import sys
import os
import json
import re

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.recommender import search

st.set_page_config(
    page_title="PaperLens - Research Paper Explorer",
    page_icon="🔎",
    layout="wide"
)

TRENDING_TOPICS = [
    "Large Language Models",
    "Diffusion Models",
    "Reinforcement Learning",
    "AI in Finance",
    "Medical AI",
]

MAX_HISTORY = 5


# ---------------- SESSION STATE ---------------- #

if "search_box" not in st.session_state:
    st.session_state.search_box = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


if st.session_state.pending_query is not None:
    st.session_state.search_box = st.session_state.pending_query
    st.session_state.pending_query = None


def set_query(value):
    st.session_state.pending_query = value


def remember(value):
    value = value.strip()
    if not value:
        return
    hist = [h for h in st.session_state.history if h.lower() != value.lower()]
    hist.insert(0, value)
    st.session_state.history = hist[:MAX_HISTORY]


def similarity_value(raw):
    """Pull a clean float % out of whatever recommender.py returns,
    so a long/unrounded float can never break the card layout."""
    match = re.search(r"[-+]?\d*\.?\d+", str(raw or ""))
    if not match:
        return 0.0
    return max(0.0, min(100.0, float(match.group())))


def score_class(value):
    if value > 90:
        return "score-high"
    if value > 80:
        return "score-mid"
    return "score-low"


# ---------------- CSS ---------------- #

with open(os.path.join(os.path.dirname(__file__), "styles.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


components.html(
    """
    <script>
    window.parent.document.addEventListener("keydown", function (e) {
        if (e.key !== "/") { return; }
        var tag = window.parent.document.activeElement.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") { return; }
        var input = window.parent.document.querySelector("input[aria-label='Search Papers']");
        if (input) {
            e.preventDefault();
            input.focus();
        }
    });
    </script>
    """,
    height=0
)


# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🔎 Paper Explorer")

    st.metric("Papers", "574,747")
    st.metric("Embeddings", "384")
    st.metric("Model", "MiniLM")
    st.metric("Search", "FAISS")

    st.markdown("**Trending Topics**")
    for topic in TRENDING_TOPICS:
        if st.button(f"🔥 {topic}", key=f"trend_{topic}"):
            set_query(topic)
            st.rerun()

    if st.session_state.history:
        st.markdown("**Recent Searches**")
        for item in st.session_state.history:
            if st.button(f"🕘 {item}", key=f"hist_{item}"):
                set_query(item)
                st.rerun()

    st.markdown(
        "<div class='sidebar-footer'>Powered by Sentence Transformers + FAISS</div>",
        unsafe_allow_html=True
    )


# ---------------- HEADER ---------------- #

st.markdown("<div class='big-title'>📚 Research Paper Explorer</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Semantic search across 574,747 AI papers.<br>"
    "Powered by Sentence Transformers and FAISS.</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='kbd-hint'>Press <kbd>/</kbd> to search</div>",
    unsafe_allow_html=True
)


# ---------------- SEARCH ---------------- #

query = st.text_input(
    "Search Papers",
    placeholder="Search research topics, methods, or papers...",
    label_visibility="collapsed",
    key="search_box",
)


# ---------------- EMPTY STATE ---------------- #

if not query:

    st.markdown(
        "<div class='empty-state'>Try one of these to get started 👇</div>",
        unsafe_allow_html=True
    )

    cols = st.columns(len(TRENDING_TOPICS))
    for col, topic in zip(cols, TRENDING_TOPICS):
        with col:
            if st.button(topic, key=f"example_{topic}", use_container_width=True):
                set_query(topic)
                st.rerun()


# ---------------- RESULTS ---------------- #

if query:

    remember(query)

    placeholder = st.empty()
    with placeholder.container():
        for _ in range(3):
            st.markdown(
                """
                <div class="skeleton-card">
                    <div class="sk-line w-50"></div>
                    <div class="sk-line w-90"></div>
                    <div class="sk-line w-70"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    start = time.time()
    results = search(query)
    elapsed = time.time() - start

    placeholder.empty()

    st.markdown(
        f"<div class='search-status'>⚡ Search completed in {elapsed:.2f} seconds</div>",
        unsafe_allow_html=True
    )
    st.write("")

    for paper in results:
        with st.container(border=True):

            st.markdown("<span class='paper-marker'></span>", unsafe_allow_html=True)

            score = similarity_value(paper["similarity"])
            badge_class = score_class(score)

            st.markdown(
                f"""
                <div class="paper-row">
                    <div class="paper-title">{paper['title']}</div>
                    <div class="score-badge {badge_class}">
                        <span class="score-dot"></span>{score:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(f"<div class='paper-authors'>{paper['authors']}</div>", unsafe_allow_html=True)

            categories = paper["categories"].split()
            badges = "".join(f"<span class='cat-pill'>{cat}</span>" for cat in categories)
            st.markdown(badges, unsafe_allow_html=True)

            abstract = paper["abstract"]
            preview = abstract if len(abstract) <= 400 else abstract[:400] + "..."
            st.markdown(f"<div class='paper-abstract'>{preview}</div>", unsafe_allow_html=True)

            title_escaped = paper["title"].replace("&", "&amp;").replace('"', "&quot;")
            components.html(
                f"""
                <html>
                <head>
                <style>
                    html, body {{ margin: 0; padding: 0; background: transparent; }}
                    #btn {{
                        background: rgba(255,255,255,0.045);
                        border: 1px solid rgba(255,255,255,0.09);
                        color: #8B93A7;
                        font-size: 12.5px;
                        font-family: sans-serif;
                        padding: 5px 12px;
                        border-radius: 8px;
                        cursor: pointer;
                    }}
                    #btn:hover {{ border-color: rgba(79,140,255,0.45); color: #E7EAF2; }}
                </style>
                </head>
                <body>
                    <button id="btn" data-title="{title_escaped}">📋 Copy title</button>
                    <script>
                        document.getElementById("btn").addEventListener("click", function() {{
                            var title = this.getAttribute("data-title");
                            navigator.clipboard.writeText(title).then(function() {{
                                document.getElementById("btn").innerText = "Copied!";
                                setTimeout(function() {{
                                    document.getElementById("btn").innerText = "Copy title";
                                }}, 1500);
                            }});
                        }});
                    </script>
                </body>
                </html>
                """,
                height=36
            )

            with st.expander("Read Full Abstract"):
                st.write(paper["abstract"])

        st.write("")