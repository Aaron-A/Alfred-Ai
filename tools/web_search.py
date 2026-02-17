"""
Web Search Tool
Search the web for current information using available search APIs.

Provider priority:
  1. Brave Search API — real web search, free tier available
  2. xAI Grok — LLM with built-in web search
  3. DuckDuckGo — instant answer API, no key required

Configure Brave: add providers.brave.api_key to alfred.json
  or set BRAVE_API_KEY env var. Get a free key at:
  https://brave.com/search/api/
"""

import json
import os
import urllib.request
import urllib.error
import urllib.parse
from core.tools import ToolRegistry, ToolParameter
from core.config import _load_config
from core.logging import get_logger

logger = get_logger("web_search")


def register(registry: ToolRegistry):
    """Register web search tools."""
    registry.register_function(
        name="web_search",
        description=(
            "Search the web for current information, news, prices, or data. "
            "Returns real search results with titles, snippets, and URLs. "
            "Use this when you need up-to-date information that isn't in your memory."
        ),
        fn=web_search,
        parameters=[
            ToolParameter("query", "string", "Search query — be specific for best results"),
            ToolParameter(
                "max_results", "integer",
                "Maximum number of results to return (default 5, max 20)",
                required=False,
            ),
        ],
        category="search",
        source="shared",
        file_path=__file__,
    )


def _get_brave_key() -> str:
    """Get Brave Search API key from config or environment."""
    cfg = _load_config()
    # Check alfred.json first
    key = cfg.get("providers", {}).get("brave", {}).get("api_key", "")
    if key:
        return key
    # Fall back to env var
    return os.getenv("BRAVE_API_KEY", "")


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using the best available provider.

    Priority: Brave Search API > xAI Grok > DuckDuckGo
    """
    max_results = min(max(1, max_results), 20)  # clamp 1-20

    # 1. Brave Search (real web search, structured results)
    brave_key = _get_brave_key()
    if brave_key:
        result = _search_via_brave(query, brave_key, max_results)
        if result:
            return result
        # If Brave failed, fall through to next provider
        logger.warning("Brave search failed, trying fallbacks")

    # 2. xAI Grok (LLM with web search capability)
    cfg = _load_config()
    providers = cfg.get("providers", {})
    xai_key = providers.get("xai", {}).get("api_key", "")
    if xai_key:
        result = _search_via_xai(query, xai_key, max_results)
        if result and "error" not in result.lower()[:50]:
            return result

    # 3. DuckDuckGo (no key needed, limited but free)
    return _search_via_ddg(query, max_results)


# ─── Brave Search ──────────────────────────────────────────

def _search_via_brave(query: str, api_key: str, max_results: int) -> str:
    """
    Brave Web Search API — real search results with titles, URLs, and snippets.
    Free tier: 2,000 queries/month. No credit card required.
    """
    params = urllib.parse.urlencode({
        "q": query,
        "count": max_results,
        "text_decorations": "false",
        "extra_snippets": "true",
    })
    url = f"https://api.search.brave.com/res/v1/web/search?{params}"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            # Handle gzip if the response is compressed
            import gzip
            raw = resp.read()
            try:
                raw = gzip.decompress(raw)
            except (gzip.BadGzipFile, OSError):
                pass  # Not gzipped, use raw bytes
            data = json.loads(raw.decode("utf-8"))

        return _format_brave_results(query, data, max_results)

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        logger.error(f"Brave search HTTP {e.code}: {body}")
        return ""  # Return empty to trigger fallback
    except Exception as e:
        logger.error(f"Brave search error: {e}")
        return ""


def _format_brave_results(query: str, data: dict, max_results: int) -> str:
    """Format Brave Search API response into readable text."""
    lines = [f"Web search results for: {query}\n"]

    # ── Infobox (quick facts, stock prices, etc.) ──
    infobox = data.get("infobox", {})
    if isinstance(infobox, dict) and infobox.get("results"):
        for box in infobox["results"][:1]:
            title = box.get("title", "")
            desc = box.get("long_desc") or box.get("description", "")
            if title or desc:
                lines.append(f"📌 **{title}**")
                if desc:
                    lines.append(f"   {desc[:300]}")
                # Key-value data (e.g. stock price, market cap)
                for attr in box.get("attributes", [])[:5]:
                    label = attr.get("label", "")
                    value = attr.get("value", "")
                    if label and value:
                        lines.append(f"   {label}: {value}")
                lines.append("")

    # ── Web results ──
    web_results = data.get("web", {}).get("results", [])
    if not web_results and not infobox:
        return f"No results found for: {query}. Try a more specific search."

    for i, result in enumerate(web_results[:max_results], 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("description", "")
        age = result.get("age", "")

        line = f"{i}. **{title}**"
        if age:
            line += f" ({age})"
        lines.append(line)

        if snippet:
            lines.append(f"   {snippet[:250]}")
        if url:
            lines.append(f"   {url}")

        # Extra snippets (more context from the page)
        extras = result.get("extra_snippets", [])
        if extras:
            lines.append(f"   > {extras[0][:200]}")

        lines.append("")

    # ── News results (if any) ──
    news = data.get("news", {}).get("results", [])
    if news:
        lines.append("📰 **Related News:**")
        for article in news[:3]:
            title = article.get("title", "")
            source = article.get("meta_url", {}).get("hostname", "")
            age = article.get("age", "")
            snippet = article.get("description", "")
            if title:
                src = f" — {source}" if source else ""
                time_str = f" ({age})" if age else ""
                lines.append(f"- {title}{src}{time_str}")
                if snippet:
                    lines.append(f"  {snippet[:150]}")
        lines.append("")

    return "\n".join(lines).strip()


# ─── xAI Grok ─────────────────────────────────────────────

def _search_via_xai(query: str, api_key: str, max_results: int) -> str:
    """Use xAI's Grok API with web search enabled. Fallback if Brave isn't configured."""
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Alfred-AI/1.0",
    }

    payload = {
        "model": "grok-3-mini-fast",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a web search assistant. Search the web and return "
                    f"the top {max_results} most relevant results for the query. "
                    "Format each result as:\n"
                    "1. **Title** — Brief summary (1-2 sentences)\n"
                    "   Source: URL\n\n"
                    "Be factual and concise. Include dates when relevant."
                ),
            },
            {"role": "user", "content": f"Search: {query}"},
        ],
        "search_parameters": {"mode": "auto"},
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return f"Web search results for: {query}\n\n{content}"
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error(f"xAI search HTTP {e.code}: {body[:200]}")
        return f"xAI search error ({e.code}): {body[:200]}"
    except Exception as e:
        logger.error(f"xAI search error: {e}")
        return f"xAI search error: {e}"


# ─── DuckDuckGo ───────────────────────────────────────────

def _search_via_ddg(query: str, max_results: int) -> str:
    """Fallback: DuckDuckGo instant answer API (limited but free, no key needed)."""
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

    headers = {"User-Agent": "Alfred-AI/1.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append(f"**{data.get('Heading', 'Answer')}** — {data['Abstract']}")
            if data.get("AbstractURL"):
                results.append(f"  Source: {data['AbstractURL']}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                text = topic["Text"][:200]
                url = topic.get("FirstURL", "")
                results.append(f"- {text}")
                if url:
                    results.append(f"  {url}")

        if not results:
            return f"No results found for: {query}. Try a more specific search."

        return f"Search results for: {query}\n(via DuckDuckGo — limited results)\n\n" + "\n".join(results)
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return f"Search unavailable: all providers failed. Configure Brave Search for best results: alfred provider add brave"
