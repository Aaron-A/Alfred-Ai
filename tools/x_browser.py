"""
X (Twitter) Browser Tool
Reply to tweets and interact with X via headless browser.

Bypasses API reply restrictions by using the web UI directly.
Uses cookies imported from Brave browser for authentication.
Cookies are auto-refreshed from Brave on each session start.
"""

import os
import re
import time
import json
from pathlib import Path
from core.tools import ToolRegistry, ToolParameter
from core.config import config
from core.logging import get_logger

logger = get_logger("x_browser")

TOOL_META = {
    "version": "0.3.0",
    "author": "Alfred AI",
    "description": "Reply to tweets and interact with X via headless browser, bypassing API restrictions",
    "dependencies": ["playwright", "browser-cookie3"],
}

# ─── State Management ─────────────────────────────────────────

_browser_state = None


def _get_state_dir() -> Path:
    """Get directory for persistent browser state."""
    state_dir = config.DATA_DIR / ".browser_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _get_state_file() -> Path:
    return _get_state_dir() / "x_session.json"


def _refresh_cookies_from_brave() -> bool:
    """Import fresh cookies from Brave browser into the session file."""
    try:
        import browser_cookie3
        cj = browser_cookie3.brave(domain_name='.x.com')
        cookies = list(cj)
        if not cookies:
            logger.warning("No X cookies found in Brave")
            return False

        pw_cookies = []
        for c in cookies:
            cookie = {
                "name": c.name,
                "value": c.value,
                "domain": c.domain,
                "path": c.path or "/",
                "secure": bool(c.secure),
                "httpOnly": False,
                "sameSite": "None" if c.secure else "Lax",
            }
            if c.expires and c.expires > 0:
                cookie["expires"] = float(c.expires)
            pw_cookies.append(cookie)

        state_file = _get_state_file()
        state_file.write_text(json.dumps({"cookies": pw_cookies, "origins": []}, indent=2))
        logger.info(f"Refreshed {len(pw_cookies)} cookies from Brave")
        return True
    except Exception as e:
        logger.warning(f"Could not refresh cookies from Brave: {e}")
        return False


def _extract_tweet_id(tweet_url: str) -> str | None:
    """Extract tweet ID from a URL or return the input if it's already an ID."""
    if tweet_url.isdigit():
        return tweet_url
    match = re.search(r'/status/(\d+)', tweet_url)
    return match.group(1) if match else None


class _XBrowserState:
    """Manages browser session for X using cookies imported from Brave."""

    def __init__(self):
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()

        state_file = _get_state_file()

        # Always try to refresh cookies from Brave for freshest session
        _refresh_cookies_from_brave()

        if not state_file.exists():
            raise RuntimeError(
                "No browser session found. Log into X in Brave browser first."
            )

        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        self._context = self._browser.new_context(
            storage_state=str(state_file),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        self._page = self._context.new_page()

        # Mask automation detection
        self._page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
        """)

        logger.info("X browser session started")

    @property
    def page(self):
        return self._page

    def save_state(self):
        """Save cookies for session persistence."""
        try:
            state_file = _get_state_file()
            self._context.storage_state(path=str(state_file))
        except Exception as e:
            logger.warning(f"Failed to save browser state: {e}")

    def close(self):
        try:
            self.save_state()
            self._context.close()
            self._browser.close()
            self._playwright.stop()
        except Exception:
            pass
        logger.info("X browser closed")


def _get_x_browser() -> _XBrowserState:
    """Get or create the X browser state."""
    global _browser_state
    if _browser_state is None:
        _browser_state = _XBrowserState()
    return _browser_state


def _close_x_browser():
    """Close the X browser."""
    global _browser_state
    if _browser_state:
        _browser_state.close()
        _browser_state = None


# ─── Tool Functions ──────────────────────────────────────────

def x_browser_reply(tweet_url: str, reply_text: str) -> str:
    """Reply to a tweet via the browser UI using the compose URL."""
    tweet_id = _extract_tweet_id(tweet_url)
    if not tweet_id:
        return f"Error: Could not extract tweet ID from: {tweet_url}"

    if len(reply_text) > 280:
        return f"Error: Reply is {len(reply_text)} chars, max 280."

    try:
        state = _get_x_browser()
    except RuntimeError as e:
        return str(e)

    page = state.page

    try:
        page.set_default_timeout(15000)

        # Use the compose URL with reply — most reliable approach
        compose_url = f"https://x.com/intent/post?in_reply_to={tweet_id}"
        page.goto(compose_url, wait_until="domcontentloaded", timeout=25000)

        # Use locators (not element handles) — they auto-retry on DOM re-renders
        text_area = page.locator('[data-testid="tweetTextarea_0"]')
        text_area.first.wait_for(state="visible", timeout=15000)

        # Check if redirected to login
        if "/login" in page.url:
            _close_x_browser()
            return (
                "Error: Not logged into X. Log into X in Brave browser, "
                "then try again (cookies are imported automatically)."
            )

        # Give X a moment to finish rendering the parent tweet preview
        time.sleep(2)

        # Remove blocking overlay divs from #layers (keeps compose dialog intact)
        _nuke_layers_js = """
            (() => {
                var layers = document.getElementById('layers');
                if (layers) {
                    Array.from(layers.children).forEach(function(child) {
                        var hasCompose = child.querySelector('[data-testid="tweetTextarea_0"]');
                        if (!hasCompose) { child.remove(); }
                    });
                }
            })();
        """
        page.evaluate(_nuke_layers_js)
        time.sleep(0.3)

        # Focus and type — use locator (auto-retries if DOM re-renders)
        text_area.first.click(force=True, timeout=5000)
        time.sleep(0.3)
        page.keyboard.insert_text(reply_text)
        time.sleep(0.5)

        # Find the Post button (locator, not element handle)
        post_btn = page.locator('[data-testid="tweetButton"], [data-testid="tweetButtonInline"]')
        post_btn.first.wait_for(state="visible", timeout=5000)

        # Check if disabled, retry typing if needed
        is_disabled = post_btn.first.evaluate(
            "el => el.hasAttribute('disabled') || el.getAttribute('aria-disabled') === 'true'"
        )
        if is_disabled:
            text_area.first.click(force=True, timeout=5000)
            page.keyboard.press("Meta+a")
            time.sleep(0.2)
            page.keyboard.insert_text(reply_text)
            time.sleep(0.5)
            is_disabled = post_btn.first.evaluate(
                "el => el.hasAttribute('disabled') || el.getAttribute('aria-disabled') === 'true'"
            )
            if is_disabled:
                return "Error: Post button is disabled. The reply text may not have been registered."

        # Remove overlays again (X may re-render #layers between type and click)
        page.evaluate(_nuke_layers_js)
        time.sleep(0.2)

        # Click post
        post_btn.first.click(timeout=10000)

        # Verify: wait for compose area to disappear (confirms post went through)
        try:
            text_area.first.wait_for(state="detached", timeout=10000)
            posted = True
        except Exception:
            posted = "compose" not in page.url and "intent" not in page.url

        state.save_state()

        if posted:
            return f"Reply posted to tweet {tweet_id}: \"{reply_text}\""
        else:
            return (
                f"Warning: Clicked Post for tweet {tweet_id} but could not confirm it went through. "
                f"The compose area is still visible — X may have blocked the reply or shown an error."
            )

    except Exception as e:
        return f"Error replying to tweet: {e}"


def x_browser_quote(tweet_url: str, quote_text: str) -> str:
    """Quote tweet via the browser UI."""
    tweet_id = _extract_tweet_id(tweet_url)
    if not tweet_id:
        return f"Error: Could not extract tweet ID from: {tweet_url}"

    if len(quote_text) > 280:
        return f"Error: Quote text is {len(quote_text)} chars, max 280."

    try:
        state = _get_x_browser()
    except RuntimeError as e:
        return str(e)

    page = state.page

    try:
        page.set_default_timeout(15000)

        # Navigate to the tweet first
        page.goto(f"https://x.com/i/status/{tweet_id}", wait_until="domcontentloaded", timeout=25000)

        if "/login" in page.url:
            _close_x_browser()
            return "Error: Not logged into X. Log into X in Brave browser first."

        # Click retweet to open menu
        rt_btn = page.locator('[data-testid="retweet"]')
        rt_btn.first.wait_for(state="visible", timeout=15000)
        rt_btn.first.click(force=True, timeout=5000)
        time.sleep(1)

        # Click Quote option
        quote_option = page.locator('a:has-text("Quote"), menuitem:has-text("Quote")')
        quote_option.first.click(timeout=5000)
        time.sleep(2)

        # Remove blocking overlays from #layers
        _nuke_layers_js = """
            (() => {
                var layers = document.getElementById('layers');
                if (layers) {
                    Array.from(layers.children).forEach(function(child) {
                        var hasCompose = child.querySelector('[data-testid="tweetTextarea_0"]');
                        if (!hasCompose) { child.remove(); }
                    });
                }
            })();
        """
        page.evaluate(_nuke_layers_js)

        # Wait for compose area
        text_area = page.locator('[data-testid="tweetTextarea_0"]')
        text_area.first.wait_for(state="visible", timeout=10000)
        text_area.first.click(force=True, timeout=5000)
        time.sleep(0.3)
        page.keyboard.insert_text(quote_text)
        time.sleep(0.5)

        # Click Post
        page.evaluate(_nuke_layers_js)
        time.sleep(0.2)
        post_btn = page.locator('[data-testid="tweetButton"], [data-testid="tweetButtonInline"]')
        post_btn.first.click(timeout=10000)

        try:
            page.wait_for_selector(
                '[data-testid="tweetTextarea_0"]', state="detached", timeout=10000
            )
            posted = True
        except Exception:
            posted = "compose" not in page.url

        state.save_state()

        if posted:
            return f"Quote tweet posted: \"{quote_text}\" quoting tweet {tweet_id}"
        else:
            return f"Warning: Clicked Post but could not confirm quote tweet went through for {tweet_id}."

    except Exception as e:
        return f"Error quoting tweet: {e}"


def x_browser_close() -> str:
    """Close the X browser session."""
    _close_x_browser()
    return "X browser session closed."


# ─── Tool Registration ──────────────────────────────────────────

def register(registry: ToolRegistry):
    """Register X browser automation tools."""

    registry.register_function(
        name="x_browser_reply",
        description=(
            "Reply to a tweet via the browser UI. Bypasses the API reply restriction "
            "that blocks replies to other users' tweets. Uses cookies from Brave browser "
            "for authentication. Pass a tweet URL or tweet ID."
        ),
        fn=x_browser_reply,
        parameters=[
            ToolParameter("tweet_url", "string",
                "Tweet URL (https://x.com/user/status/123) or tweet ID"),
            ToolParameter("reply_text", "string",
                "The reply text (max 280 chars)"),
        ],
        category="social",
        source="shared",
        file_path=__file__,
        dependencies=["playwright", "browser-cookie3"],
    )

    registry.register_function(
        name="x_browser_quote",
        description=(
            "Quote tweet via the browser UI. Bypasses API quote restrictions. "
            "Uses cookies from Brave browser. Pass a tweet URL or tweet ID."
        ),
        fn=x_browser_quote,
        parameters=[
            ToolParameter("tweet_url", "string",
                "Tweet URL (https://x.com/user/status/123) or tweet ID"),
            ToolParameter("quote_text", "string",
                "Your commentary text (max 280 chars)"),
        ],
        category="social",
        source="shared",
        file_path=__file__,
        dependencies=["playwright", "browser-cookie3"],
    )

    registry.register_function(
        name="x_browser_close",
        description="Close the X browser session and free resources.",
        fn=x_browser_close,
        parameters=[],
        category="social",
        source="shared",
        file_path=__file__,
        dependencies=["playwright", "browser-cookie3"],
    )
