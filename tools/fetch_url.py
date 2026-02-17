"""
Fetch URL Tool
Retrieve and extract readable text content from web pages.

Complements web_search — search finds URLs, this tool reads them.
Uses urllib (stdlib) with a real User-Agent to avoid bot detection.
HTML is stripped to plain text; output is truncated to keep LLM context lean.
"""

import json
import re
import urllib.request
import urllib.error
from core.tools import ToolRegistry, ToolParameter
from core.logging import get_logger

logger = get_logger("fetch_url")


def register(registry: ToolRegistry):
    """Register the fetch_url tool."""
    registry.register_function(
        name="fetch_url",
        description=(
            "Fetch a web page and return its text content. "
            "Use this to read articles, documentation, API responses, or any URL. "
            "HTML is automatically stripped to plain readable text. "
            "Output is truncated to ~8000 chars to stay within context limits."
        ),
        fn=fetch_url,
        parameters=[
            ToolParameter("url", "string", "The URL to fetch (must start with http:// or https://)"),
            ToolParameter(
                "max_chars", "integer",
                "Maximum characters to return (default 8000, max 20000)",
                required=False,
            ),
        ],
        category="web",
        source="shared",
        file_path=__file__,
    )


def fetch_url(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return its text content."""
    max_chars = min(max(500, max_chars), 20000)

    # Validate URL
    if not url.startswith(("http://", "https://")):
        return f"Error: URL must start with http:// or https:// (got: {url[:50]})"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get("Content-Type", "")

            # Handle gzip
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                try:
                    raw = gzip.decompress(raw)
                except (gzip.BadGzipFile, OSError):
                    pass

            text = raw.decode("utf-8", errors="replace")

        # If it's JSON, format it nicely
        if "json" in content_type or text.strip().startswith("{"):
            try:
                data = json.loads(text)
                formatted = json.dumps(data, indent=2, ensure_ascii=False)
                if len(formatted) > max_chars:
                    formatted = formatted[:max_chars] + "\n... (truncated)"
                return f"JSON from {url}:\n\n{formatted}"
            except (json.JSONDecodeError, ValueError):
                pass

        # If it's plain text (not HTML), return directly
        if "text/plain" in content_type or not _looks_like_html(text):
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"
            return text

        # Strip HTML to readable text
        readable = _html_to_text(text)

        if not readable.strip():
            return f"Page at {url} returned no readable text content."

        if len(readable) > max_chars:
            readable = readable[:max_chars] + "\n... (truncated)"

        return readable

    except urllib.error.HTTPError as e:
        return f"HTTP Error {e.code}: {e.reason} — {url}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason} — {url}"
    except TimeoutError:
        return f"Timeout: page took too long to respond — {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def _looks_like_html(text: str) -> bool:
    """Quick check if content looks like HTML."""
    start = text[:500].lower()
    return "<html" in start or "<!doctype" in start or "<head" in start


def _html_to_text(html: str) -> str:
    """
    Strip HTML tags and extract readable text.

    Simple but effective — no external dependencies (no BeautifulSoup needed).
    Handles the common cases: scripts, styles, block elements, entities.
    """
    # Remove script and style blocks entirely
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # Convert common block elements to newlines
    html = re.sub(r"<(br|hr)[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</(p|div|h[1-6]|li|tr|blockquote|section|article|header|footer)>",
                  "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<(p|div|h[1-6]|li|tr|blockquote|section|article|header|footer)[^>]*>",
                  "\n", html, flags=re.IGNORECASE)

    # Strip remaining tags
    html = re.sub(r"<[^>]+>", "", html)

    # Decode common HTML entities
    html = html.replace("&amp;", "&")
    html = html.replace("&lt;", "<")
    html = html.replace("&gt;", ">")
    html = html.replace("&quot;", '"')
    html = html.replace("&#39;", "'")
    html = html.replace("&nbsp;", " ")
    html = html.replace("&mdash;", "—")
    html = html.replace("&ndash;", "–")
    html = html.replace("&rsquo;", "'")
    html = html.replace("&lsquo;", "'")
    html = html.replace("&rdquo;", "\u201d")
    html = html.replace("&ldquo;", "\u201c")
    # Numeric entities
    html = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), html)
    html = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), html)

    # Collapse whitespace
    lines = []
    for line in html.split("\n"):
        cleaned = " ".join(line.split())
        if cleaned:
            lines.append(cleaned)

    # Collapse multiple blank lines
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
