"""
HTTP Request Tool
Make HTTP requests to APIs and return the response.

For structured API interactions — GET, POST, PUT, DELETE with headers and body.
Unlike fetch_url (which is for reading web pages), this is for talking to APIs.
"""

import json
import urllib.request
import urllib.error
import urllib.parse
from core.tools import ToolRegistry, ToolParameter
from core.config import config
from core.logging import get_logger

logger = get_logger("http_request")

# Block requests to known sensitive/internal endpoints
_BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "metadata.google.internal",          # GCP metadata
    "169.254.169.254",                   # AWS/Azure metadata
}


# Service-specific auth header injection
_SERVICE_AUTH_MAP = {
    "alpaca": lambda headers, creds: headers.update({
        "APCA-API-KEY-ID": creds.get("api_key", ""),
        "APCA-API-SECRET-KEY": creds.get("secret_key", ""),
    }),
}


def _inject_service_auth(headers: dict, service_name: str, creds: dict):
    """Inject authentication headers for a known service."""
    injector = _SERVICE_AUTH_MAP.get(service_name)
    if injector:
        injector(headers, creds)
        logger.debug(f"Auto-injected auth headers for service: {service_name}")


def register(registry: ToolRegistry):
    """Register HTTP request tools."""
    registry.register_function(
        name="http_request",
        description=(
            "Make an HTTP request to an API endpoint. "
            "Supports GET, POST, PUT, DELETE with custom headers and JSON body. "
            "Use this for API calls, webhooks, or any structured HTTP interaction. "
            "For reading web pages, use fetch_url instead."
        ),
        fn=http_request,
        parameters=[
            ToolParameter("url", "string", "The URL to request"),
            ToolParameter("method", "string", "HTTP method: GET, POST, PUT, DELETE (default: GET)", required=False),
            ToolParameter("headers", "string", "JSON string of headers, e.g. '{\"Authorization\": \"Bearer xxx\"}'", required=False),
            ToolParameter("body", "string", "Request body — JSON string for POST/PUT", required=False),
            ToolParameter("timeout", "integer", "Timeout in seconds (default 30, max 60)", required=False),
        ],
        category="web",
        source="shared",
        file_path=__file__,
    )


def http_request(
    url: str,
    method: str = "GET",
    headers: str = None,
    body: str = None,
    timeout: int = 30,
) -> str:
    """Make an HTTP request and return the response."""
    timeout = min(max(5, timeout), 60)
    method = method.upper()

    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"):
        return f"Error: Unsupported HTTP method '{method}'. Use GET, POST, PUT, DELETE, PATCH, or HEAD."

    # Validate URL
    if not url.startswith(("http://", "https://")):
        return f"Error: URL must start with http:// or https://"

    # Block internal/metadata endpoints
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
        if host in _BLOCKED_HOSTS:
            return f"Error: Requests to {host} are blocked for security."
    except Exception:
        return f"Error: Invalid URL format."

    # Parse headers
    req_headers = {
        "User-Agent": "Alfred-AI/1.0",
        "Accept": "application/json, text/plain, */*",
    }
    if headers:
        try:
            custom = json.loads(headers)
            if isinstance(custom, dict):
                req_headers.update(custom)
            else:
                return "Error: headers must be a JSON object, e.g. '{\"Key\": \"Value\"}'"
        except json.JSONDecodeError as e:
            return f"Error parsing headers JSON: {e}"

    # Auto-inject auth headers for configured services
    try:
        domain_map = config.get_service_domains()
        if host in domain_map:
            svc = domain_map[host]
            svc_name = svc.get("service", "")
            _inject_service_auth(req_headers, svc_name, svc)
    except Exception as e:
        logger.warning(f"Service auth injection failed: {e}")

    # Prepare body
    data = None
    if body and method in ("POST", "PUT", "PATCH"):
        if "Content-Type" not in req_headers:
            # Auto-detect: if body looks like JSON, set content-type
            try:
                json.loads(body)
                req_headers["Content-Type"] = "application/json"
            except (json.JSONDecodeError, ValueError):
                req_headers["Content-Type"] = "text/plain"
        data = body.encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            resp_headers = dict(resp.headers)
            raw = resp.read()

            # Handle gzip
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                try:
                    raw = gzip.decompress(raw)
                except (gzip.BadGzipFile, OSError):
                    pass

            text = raw.decode("utf-8", errors="replace")

        # Format response
        lines = [f"HTTP {status}"]

        # Try to pretty-print JSON
        content_type = resp_headers.get("Content-Type", "")
        if "json" in content_type or _looks_like_json(text):
            try:
                data = json.loads(text)
                text = json.dumps(data, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                pass

        # Truncate if too long
        if len(text) > 10000:
            text = text[:10000] + "\n... (truncated)"

        lines.append(text)
        return "\n".join(lines)

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
        return f"Request timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def _looks_like_json(text: str) -> bool:
    """Quick check if text looks like JSON."""
    stripped = text.strip()
    return (stripped.startswith("{") and stripped.endswith("}")) or \
           (stripped.startswith("[") and stripped.endswith("]"))
