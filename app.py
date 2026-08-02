import os
import requests
import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS

# Load API key
load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY") or st.secrets.get("FIREWORKS_API_KEY")
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
TEXT_MODEL = "accounts/fireworks/models/glm-5p2"  # text LLM (from TrustZero)


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Query DuckDuckGo and return list of {title, url, body}.
    Returns empty list on failure.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        # Normalize keys (duckduckgo_search uses 'body' for snippet)
        return [
            {"title": r.get("title", ""), "url": r.get("href", r.get("url", "")),
             "body": r.get("body", r.get("snippet", ""))}
            for r in results
        ]
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.error(f"DuckDuckGo search failed: {e}")
        return []


def synthesise(query: str, sources: list[dict], api_key: str) -> str:
    """
    Send query + sources to Fireworks text LLM.
    Returns synthesized answer with [N] citations.
    """
    context = "\n".join(
        f"[{i+1}] {s['title']}\n{s['body']}"
        for i, s in enumerate(sources)
    )
    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {"role": "system",
             "content": "Answer using only the sources below. "
                        "Cite sources with [N] markers like [1], [2]. "
                        "If the sources don't contain the answer, say 'Insufficient information in sources.'"},
            {"role": "user",
             "content": f"Query: {query}\n\nSources:\n{context}"}
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }
    r = requests.post(
        FIREWORKS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    if r.status_code == 401:
        raise RuntimeError("Invalid or missing Fireworks API key (HTTP 401).")
    if r.status_code == 429:
        raise RuntimeError("Fireworks rate limit or insufficient credit (HTTP 429).")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ───────── Streamlit UI ─────────
st.set_page_config(page_title="AI Search Engine", page_icon="🔍")
st.title("🔍 AI-Powered Search Engine")
st.caption("Ask a question. We search the web, then an LLM synthesizes an answer with citations.")

query = st.text_input("Your question", placeholder="e.g., What is the latest version of Python?")

if st.button("Search") and query.strip():
    if not FIREWORKS_API_KEY:
        st.error("FIREWORKS_API_KEY is missing. Add it to your .env file.")
    else:
        # Step 1: Retrieve from DuckDuckGo
        with st.spinner("Searching the web..."):
            sources = search_web(query, max_results=5)

        if not sources:
            st.warning("No search results found. Try a different query.")
        else:
            # Show sources
            st.subheader("Sources")
            for i, s in enumerate(sources):
                st.markdown(f"**[{i+1}]** [{s['title']}]({s['url']})")
                st.caption(s['body'])

            # Step 2: Augment + Generate
            with st.spinner("Synthesizing answer with LLM..."):
                try:
                    answer = synthesise(query, sources, FIREWORKS_API_KEY)
                    st.subheader("Answer")
                    st.write(answer)
                except RuntimeError as e:
                    st.error(str(e))
                    # Fallback: show raw results without synthesis
                    st.info("Showing raw search results (LLM synthesis failed).")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.info("Showing raw search results (LLM synthesis failed).")