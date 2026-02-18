"""
X (Twitter) API Tool
Authenticated access to X API v2 with OAuth 1.0a signing.

Handles all the OAuth complexity internally — agents just call the tool
with an endpoint and get back authenticated responses. No credentials
in workspace files, no manual headers.

Uses only Python stdlib: hmac, hashlib, base64, urllib, uuid, time.
"""

import hmac
import hashlib
import base64
import time
import uuid
import json
import urllib.request
import urllib.error
import urllib.parse
from core.tools import ToolRegistry, ToolParameter
from core.config import config
from core.logging import get_logger

logger = get_logger("x_api")

X_API_BASE = "https://api.x.com"


# ─── OAuth 1.0a Signing ─────────────────────────────────────────

def _percent_encode(s: str) -> str:
    """RFC 3986 percent-encoding (required by OAuth 1.0a)."""
    return urllib.parse.quote(str(s), safe="")


def _generate_oauth_signature(
    method: str,
    url: str,
    oauth_params: dict,
    query_params: dict,
    consumer_secret: str,
    token_secret: str,
) -> str:
    """
    Generate OAuth 1.0a HMAC-SHA1 signature.

    1. Collect ALL parameters (oauth_* + query params)
    2. Sort alphabetically by key, then by value
    3. Percent-encode each key and value
    4. Join as key=value with &
    5. Build signature base string: METHOD&encode(url)&encode(param_string)
    6. Build signing key: encode(consumer_secret)&encode(token_secret)
    7. HMAC-SHA1(signing_key, base_string)
    8. Base64 encode
    """
    # Collect and sort all params
    all_params = {}
    all_params.update(oauth_params)
    all_params.update(query_params)

    sorted_params = sorted(all_params.items(), key=lambda x: (x[0], x[1]))

    # Build parameter string
    param_string = "&".join(
        f"{_percent_encode(k)}={_percent_encode(str(v))}"
        for k, v in sorted_params
    )

    # Build signature base string
    base_string = "&".join([
        method.upper(),
        _percent_encode(url),
        _percent_encode(param_string),
    ])

    # Build signing key
    signing_key = f"{_percent_encode(consumer_secret)}&{_percent_encode(token_secret)}"

    # HMAC-SHA1 and Base64
    hashed = hmac.new(
        signing_key.encode("utf-8"),
        base_string.encode("utf-8"),
        hashlib.sha1,
    )
    return base64.b64encode(hashed.digest()).decode("utf-8")


def _build_oauth_header(
    method: str,
    url: str,
    query_params: dict,
    creds: dict,
) -> str:
    """
    Build the full OAuth Authorization header value.

    Returns: 'OAuth oauth_consumer_key="...", oauth_nonce="...", ...'
    """
    oauth_params = {
        "oauth_consumer_key": creds["api_key"],
        "oauth_nonce": uuid.uuid4().hex,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": creds["access_token"],
        "oauth_version": "1.0",
    }

    signature = _generate_oauth_signature(
        method=method,
        url=url,
        oauth_params=oauth_params,
        query_params=query_params,
        consumer_secret=creds["api_secret"],
        token_secret=creds["access_token_secret"],
    )
    oauth_params["oauth_signature"] = signature

    auth_parts = ", ".join(
        f'{_percent_encode(k)}="{_percent_encode(v)}"'
        for k, v in sorted(oauth_params.items())
    )
    return f"OAuth {auth_parts}"


def _get_x_credentials() -> dict:
    """Load X credentials from alfred.json services.x section."""
    return config.get_service_credentials("x")


def _get_user_id(creds: dict) -> str:
    """Get the authenticated user's ID, fetching from API if needed."""
    uid = creds.get("user_id", "")
    if uid:
        return uid

    # Fetch from API
    result = x_api_request(endpoint="/2/users/me")
    try:
        body = result.split("\n", 1)[1]
        data = json.loads(body)
        uid = data["data"]["id"]
        logger.info(f"Resolved X user ID: {uid}")
        # Try to persist it for next time
        try:
            from core.config import _load_config, _save_config
            cfg = _load_config()
            if "x" in cfg.get("services", {}):
                cfg["services"]["x"]["user_id"] = uid
                _save_config(cfg)
                logger.info("Saved user_id to alfred.json")
        except Exception as e:
            logger.warning(f"Could not persist user_id: {e}")
        return uid
    except (json.JSONDecodeError, KeyError, IndexError):
        return ""


# ─── Core Request Function ──────────────────────────────────────

def x_api_request(
    endpoint: str,
    method: str = "GET",
    body: str = None,
    params: str = None,
) -> str:
    """Make an authenticated X API v2 request with OAuth 1.0a signing."""
    method = method.upper()
    creds = _get_x_credentials()

    if not creds.get("api_key"):
        return "Error: X API credentials not configured. Run: alfred service add x"

    # Build full URL
    if endpoint.startswith("http"):
        url = endpoint
    else:
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        url = f"{X_API_BASE}{endpoint}"

    # Parse query params
    query_params = {}
    if params:
        try:
            query_params = json.loads(params)
        except json.JSONDecodeError as e:
            return f"Error parsing params JSON: {e}"

    # Build URL with query params for the actual HTTP request
    full_url = url
    if query_params and method == "GET":
        qs = urllib.parse.urlencode(query_params, quote_via=urllib.parse.quote)
        full_url = f"{url}?{qs}"

    # OAuth signature uses base URL (without query string)
    # Query params are included in the signature via the params dict
    auth_header = _build_oauth_header(method, url, query_params if method == "GET" else {}, creds)

    headers = {
        "Authorization": auth_header,
        "User-Agent": "Alfred-AI/1.0",
        "Content-Type": "application/json",
    }

    # Prepare body for POST/PUT/PATCH
    data = None
    if body and method in ("POST", "PUT", "PATCH"):
        data = body.encode("utf-8")

    req = urllib.request.Request(full_url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8", errors="replace")

            # Pretty-print JSON
            try:
                parsed = json.loads(raw)
                raw = json.dumps(parsed, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                pass

            # Truncate if too long
            if len(raw) > 10000:
                raw = raw[:10000] + "\n... (truncated)"

            return f"HTTP {status}\n{raw}"

    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            pass
        result = f"HTTP {e.code} {e.reason}"
        if body_text:
            result += f"\n{body_text}"
        return result
    except urllib.error.URLError as e:
        return f"Connection error: {e.reason}"
    except TimeoutError:
        return "Request timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


# ─── Convenience Functions ───────────────────────────────────────

def x_post_tweet(text: str, reply_to: str = None) -> str:
    """Post a tweet or reply to X/Twitter."""
    if len(text) > 280:
        return f"Error: Tweet is {len(text)} chars, max 280."

    # Guard: if text starts with @username but no reply_to, the post will be a
    # standalone mention (not a threaded reply). Warn the caller so the LLM can
    # retry with the correct reply_to tweet ID.
    if text.lstrip().startswith("@") and not reply_to:
        return (
            "Error: Text starts with @username but no reply_to tweet ID was provided. "
            "Without reply_to, this will post as a standalone mention — NOT a threaded reply. "
            "Please call again with the reply_to parameter set to the tweet ID you want to reply to."
        )

    body = {"text": text}
    if reply_to:
        body["reply"] = {"in_reply_to_tweet_id": str(reply_to)}

    result = x_api_request(
        endpoint="/2/tweets",
        method="POST",
        body=json.dumps(body),
    )

    # Parse the response to extract the tweet ID for convenience
    try:
        lines = result.split("\n", 1)
        if lines[0].startswith("HTTP 2"):
            data = json.loads(lines[1])
            tweet_id = data.get("data", {}).get("id", "?")
            action = "Reply" if reply_to else "Tweet"
            tweet_url = f"https://x.com/DarkStoneCap/status/{tweet_id}"
            return f"{action} posted successfully.\nLink (copy this exactly): {tweet_url}\n{result}"
    except (json.JSONDecodeError, IndexError, KeyError):
        pass

    return result


def x_like_tweet(tweet_id: str) -> str:
    """Like a tweet on X/Twitter."""
    creds = _get_x_credentials()
    user_id = _get_user_id(creds)
    if not user_id:
        return "Error: Could not determine your X user ID."

    return x_api_request(
        endpoint=f"/2/users/{user_id}/likes",
        method="POST",
        body=json.dumps({"tweet_id": str(tweet_id)}),
    )


def x_follow_user(username: str) -> str:
    """Follow a user on X/Twitter by username."""
    username = username.lstrip("@")

    # Look up target user ID
    lookup = x_api_request(
        endpoint=f"/2/users/by/username/{username}",
    )
    try:
        data = json.loads(lookup.split("\n", 1)[1])
        target_id = data["data"]["id"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return f"Error: Could not find user @{username}. Response: {lookup}"

    # Get own user ID
    creds = _get_x_credentials()
    user_id = _get_user_id(creds)
    if not user_id:
        return "Error: Could not determine your X user ID."

    return x_api_request(
        endpoint=f"/2/users/{user_id}/following",
        method="POST",
        body=json.dumps({"target_user_id": target_id}),
    )


def x_search_tweets(query: str, max_results: int = 10) -> str:
    """Search recent tweets on X/Twitter."""
    max_results = min(max(10, max_results), 100)
    return x_api_request(
        endpoint="/2/tweets/search/recent",
        params=json.dumps({
            "query": query,
            "max_results": str(max_results),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "name,username,verified",
        }),
    )


def x_get_metrics() -> str:
    """Get your X/Twitter account metrics for monetization tracking."""
    return x_api_request(
        endpoint="/2/users/me",
        params=json.dumps({
            "user.fields": "public_metrics,verified,description,created_at",
        }),
    )


# ─── Tool Registration ──────────────────────────────────────────

def register(registry: ToolRegistry):
    """Register X (Twitter) API tools."""

    registry.register_function(
        name="x_api",
        description=(
            "Make an authenticated request to the X (Twitter) API v2. "
            "Handles OAuth 1.0a signing automatically. Use this for any "
            "X API endpoint not covered by the convenience tools."
        ),
        fn=x_api_request,
        parameters=[
            ToolParameter("endpoint", "string",
                "API endpoint path, e.g. '/2/tweets' or '/2/users/me'"),
            ToolParameter("method", "string",
                "HTTP method: GET, POST, DELETE (default: GET)", required=False),
            ToolParameter("body", "string",
                "JSON request body for POST requests", required=False),
            ToolParameter("params", "string",
                "JSON string of query parameters for GET requests", required=False),
        ],
        category="social",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="x_post_tweet",
        description=(
            "Post a tweet or reply to X/Twitter. Max 280 characters. "
            "IMPORTANT: To reply to a tweet, you MUST provide the reply_to parameter "
            "with the tweet ID. Without reply_to, the post is a standalone tweet — "
            "starting text with @username alone does NOT make it a reply."
        ),
        fn=x_post_tweet,
        parameters=[
            ToolParameter("text", "string", "The tweet text (max 280 chars)"),
            ToolParameter("reply_to", "string",
                "Tweet ID to reply to. REQUIRED for replies — without this, "
                "the tweet will NOT appear as a reply in the thread, even if "
                "text starts with @username. Omit only for original tweets.",
                required=False),
        ],
        category="social",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="x_like_tweet",
        description="Like a tweet on X/Twitter.",
        fn=x_like_tweet,
        parameters=[
            ToolParameter("tweet_id", "string", "The tweet ID to like"),
        ],
        category="social",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="x_follow_user",
        description="Follow a user on X/Twitter by their username.",
        fn=x_follow_user,
        parameters=[
            ToolParameter("username", "string", "The username to follow (without @)"),
        ],
        category="social",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="x_search_tweets",
        description=(
            "Search recent tweets on X/Twitter. Supports X API v2 query syntax: "
            "from:username, to:username, #hashtag, keyword, etc."
        ),
        fn=x_search_tweets,
        parameters=[
            ToolParameter("query", "string",
                "Search query (e.g. 'from:elonmusk', '#SpaceX', 'AI agents')"),
            ToolParameter("max_results", "integer",
                "Number of results 10-100 (default: 10)", required=False),
        ],
        category="social",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="x_get_metrics",
        description=(
            "Get your X/Twitter account metrics: followers, following, "
            "tweet count, listed count. Use for monetization tracking."
        ),
        fn=x_get_metrics,
        parameters=[],
        category="social",
        source="shared",
        file_path=__file__,
    )
