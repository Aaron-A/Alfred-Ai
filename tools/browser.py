"""
Browser Automation Tool
Headless browser for navigating, clicking, filling, screenshotting, and scraping web pages.

Uses Playwright (sync API) with a singleton browser that persists across tool calls,
so agents can navigate → click → fill → read in sequence within a session.
Auto-closes after 5 minutes of inactivity.
"""

import os
import time
import json
import ipaddress
import socket
import urllib.parse
from pathlib import Path
from datetime import datetime
from core.tools import ToolRegistry, ToolParameter
from core.config import config
from core.logging import get_logger

logger = get_logger("browser")

TOOL_META = {
    "version": "0.1.0",
    "author": "Alfred AI",
    "description": "Headless browser automation — navigate, click, fill, screenshot, and extract text from web pages",
    "dependencies": ["playwright"],
}

MAX_TEXT_CHARS = 8000
INACTIVITY_TIMEOUT = 300  # 5 minutes

# ─── Security ────────────────────────────────────────────────────

_BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "metadata.google.internal",
    "169.254.169.254",
}

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
]


def _is_url_blocked(url: str) -> tuple[bool, str]:
    """Check if a URL targets a blocked internal/metadata endpoint."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
        if host in _BLOCKED_HOSTS:
            return True, f"Navigation to {host} is blocked for security."
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
            for network in _BLOCKED_NETWORKS:
                if ip in network:
                    return True, f"Navigation to private network ({ip}) is blocked."
        except (socket.gaierror, ValueError):
            pass
    except Exception:
        return True, "Invalid URL format."
    return False, ""


# ─── Browser State (Singleton) ──────────────────────────────────

_browser_state = None


class _BrowserState:
    """Manages a singleton Playwright browser instance across tool calls."""

    def __init__(self, headless: bool = True):
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        self._context = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        self._page = self._context.new_page()
        self._last_activity: float = time.time()
        logger.info("Browser launched (headless=%s)", headless)

    @property
    def page(self):
        self._last_activity = time.time()
        return self._page

    def is_stale(self) -> bool:
        return (time.time() - self._last_activity) > INACTIVITY_TIMEOUT

    def close(self):
        try:
            self._context.close()
            self._browser.close()
            self._playwright.stop()
        except Exception:
            pass
        logger.info("Browser closed")


def _get_browser(headless: bool = True) -> _BrowserState:
    """Get or create the singleton browser state."""
    global _browser_state
    if _browser_state is not None and _browser_state.is_stale():
        logger.info("Browser auto-closed after inactivity timeout")
        _browser_state.close()
        _browser_state = None
    if _browser_state is None:
        _browser_state = _BrowserState(headless=headless)
    return _browser_state


def _get_browser_or_error():
    """Get browser state, returning error string if not initialized."""
    global _browser_state
    if _browser_state is None:
        return "Error: No browser is open. Call browser_open(url) first."
    if _browser_state.is_stale():
        _browser_state.close()
        _browser_state = None
        return "Error: Browser was closed due to inactivity. Call browser_open(url) to reopen."
    return _browser_state


def _truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... (truncated, {len(text)} chars total)"
    return text


def _get_screenshot_dir() -> Path:
    """Get screenshot directory, preferring agent workspace."""
    ws = os.environ.get("ALFRED_WORKSPACE", "")
    if ws:
        ss_dir = Path(ws) / "screenshots"
    else:
        ss_dir = config.DATA_DIR / "screenshots"
    ss_dir.mkdir(parents=True, exist_ok=True)
    return ss_dir


# ─── Tool Functions ──────────────────────────────────────────────

def browser_open(url: str, wait_for: str = None) -> str:
    """Navigate to a URL and return page title + text summary."""
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    blocked, reason = _is_url_blocked(url)
    if blocked:
        return f"Error: {reason}"

    try:
        state = _get_browser()
        page = state.page
        page.goto(url, timeout=30000, wait_until="domcontentloaded")

        if wait_for:
            try:
                page.wait_for_selector(wait_for, timeout=10000)
            except Exception:
                pass  # Continue even if selector not found

        title = page.title()
        text = page.inner_text("body") or ""
        text = _truncate(text, 4000)

        return f"Page loaded: {title}\nURL: {page.url}\n\n{text}"
    except Exception as e:
        return f"Error opening {url}: {e}"


def browser_screenshot(filename: str = None, full_page: bool = False) -> str:
    """Take a screenshot of the current page."""
    state = _get_browser_or_error()
    if isinstance(state, str):
        return state

    ss_dir = _get_screenshot_dir()
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
    if not filename.endswith(".png"):
        filename += ".png"

    filepath = ss_dir / filename
    try:
        state.page.screenshot(path=str(filepath), full_page=full_page)
        return f"Screenshot saved: {filepath}"
    except Exception as e:
        return f"Error taking screenshot: {e}"


def browser_click(selector: str) -> str:
    """Click an element by CSS selector."""
    state = _get_browser_or_error()
    if isinstance(state, str):
        return state

    try:
        state.page.click(selector, timeout=10000)
        try:
            state.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        title = state.page.title()
        return f"Clicked '{selector}'. Page: {title} ({state.page.url})"
    except Exception as e:
        return f"Error clicking '{selector}': {e}"


def browser_fill(selector: str, value: str) -> str:
    """Fill an input field by CSS selector."""
    state = _get_browser_or_error()
    if isinstance(state, str):
        return state

    try:
        state.page.fill(selector, value, timeout=10000)
        return f"Filled '{selector}' with {len(value)} chars."
    except Exception as e:
        return f"Error filling '{selector}': {e}"


def browser_text(selector: str = None, max_chars: int = 8000) -> str:
    """Extract text from page or element."""
    state = _get_browser_or_error()
    if isinstance(state, str):
        return state

    max_chars = min(max(500, max_chars), 20000)

    try:
        if selector:
            text = state.page.inner_text(selector, timeout=5000)
        else:
            text = state.page.inner_text("body")

        text = text.strip()
        if not text:
            return "No text content found."

        return _truncate(text, max_chars)
    except Exception as e:
        return f"Error extracting text: {e}"


def browser_eval(js: str) -> str:
    """Execute JavaScript in the page context."""
    state = _get_browser_or_error()
    if isinstance(state, str):
        return state

    if len(js) > 10000:
        return "Error: JavaScript code too long (max 10000 chars)."

    try:
        result = state.page.evaluate(js)
        if result is None:
            return "(no return value)"
        if isinstance(result, str):
            return _truncate(result)
        return _truncate(json.dumps(result, indent=2, default=str))
    except Exception as e:
        return f"Error executing JavaScript: {e}"


def browser_close() -> str:
    """Close the browser and free resources."""
    global _browser_state
    if _browser_state is None:
        return "No browser is open."

    try:
        _browser_state.close()
    except Exception:
        pass
    _browser_state = None
    return "Browser closed."


# ─── Tool Registration ──────────────────────────────────────────

def register(registry: ToolRegistry):
    """Register browser automation tools."""

    registry.register_function(
        name="browser_open",
        description=(
            "Open a URL in a headless browser and return the page title and text summary. "
            "The browser stays open between calls — use browser_click, browser_fill, "
            "browser_text to interact with the page. Use browser_close when done."
        ),
        fn=browser_open,
        parameters=[
            ToolParameter("url", "string",
                "URL to navigate to (must start with http:// or https://)"),
            ToolParameter("wait_for", "string",
                "CSS selector to wait for before returning (optional)",
                required=False),
        ],
        category="browser",
        source="shared",
        file_path=__file__,
        dependencies=["playwright"],
    )

    registry.register_function(
        name="browser_screenshot",
        description=(
            "Take a screenshot of the current browser page. "
            "Saves to the agent's workspace. Returns the file path. "
            "Call browser_open first."
        ),
        fn=browser_screenshot,
        parameters=[
            ToolParameter("filename", "string",
                "Filename for screenshot (default: auto-generated timestamp)",
                required=False),
            ToolParameter("full_page", "boolean",
                "Capture full scrollable page (default: false, viewport only)",
                required=False),
        ],
        category="browser",
        source="shared",
        file_path=__file__,
        dependencies=["playwright"],
    )

    registry.register_function(
        name="browser_click",
        description=(
            "Click an element on the current page by CSS selector. "
            "Returns confirmation or error. Call browser_open first."
        ),
        fn=browser_click,
        parameters=[
            ToolParameter("selector", "string",
                "CSS selector of element to click (e.g. 'button.submit', '#login')"),
        ],
        category="browser",
        source="shared",
        file_path=__file__,
        dependencies=["playwright"],
    )

    registry.register_function(
        name="browser_fill",
        description=(
            "Fill an input field on the current page. "
            "Clears existing content and types the new value."
        ),
        fn=browser_fill,
        parameters=[
            ToolParameter("selector", "string",
                "CSS selector of input field (e.g. 'input[name=\"username\"]', '#search-box')"),
            ToolParameter("value", "string", "Text to type into the field"),
        ],
        category="browser",
        source="shared",
        file_path=__file__,
        dependencies=["playwright"],
    )

    registry.register_function(
        name="browser_text",
        description=(
            "Extract text content from the current page or a specific element. "
            "Without a selector, returns full page text. "
            "With a selector, returns text from that element only."
        ),
        fn=browser_text,
        parameters=[
            ToolParameter("selector", "string",
                "CSS selector to extract text from (optional, default: full page)",
                required=False),
            ToolParameter("max_chars", "integer",
                "Maximum characters to return (default 8000, max 20000)",
                required=False),
        ],
        category="browser",
        source="shared",
        file_path=__file__,
        dependencies=["playwright"],
    )

    registry.register_function(
        name="browser_eval",
        description=(
            "Execute JavaScript in the current browser page and return the result. "
            "Use for advanced interactions, reading page state, or extracting structured data."
        ),
        fn=browser_eval,
        parameters=[
            ToolParameter("js", "string",
                "JavaScript code to execute in the page context"),
        ],
        category="browser",
        source="shared",
        file_path=__file__,
        dependencies=["playwright"],
    )

    registry.register_function(
        name="browser_close",
        description=(
            "Close the browser and free resources. "
            "Call when done browsing. Also auto-closes after 5 minutes of inactivity."
        ),
        fn=browser_close,
        parameters=[],
        category="browser",
        source="shared",
        file_path=__file__,
        dependencies=["playwright"],
    )
