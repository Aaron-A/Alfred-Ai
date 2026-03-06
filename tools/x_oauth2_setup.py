#!/usr/bin/env python3
"""
OAuth 2.0 PKCE Setup for X (Twitter) API

Run this once to authorize @DarkStoneCap and get OAuth 2.0 user-context tokens.
These tokens allow posting, replying, liking, etc. with the same permissions
as the web UI — bypassing the reply restrictions that affect OAuth 1.0a.

Usage:
    python tools/x_oauth2_setup.py
    python tools/x_oauth2_setup.py --client-id YOUR_CLIENT_ID

The Client ID is found in X Developer Portal > App > Keys and tokens > OAuth 2.0
"""

import hashlib
import secrets
import base64
import json
import urllib.request
import urllib.parse
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import webbrowser
import sys
import os
import time

# X OAuth 2.0 endpoints
AUTH_URL = "https://twitter.com/i/oauth2/authorize"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
REDIRECT_URI = "http://localhost:3000"
SCOPES = "tweet.read tweet.write users.read offline.access"


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alfred.json")
    with open(config_path) as f:
        return json.load(f), config_path


def save_config(cfg, config_path):
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


def generate_pkce():
    """Generate PKCE code_verifier and code_challenge (S256)."""
    code_verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler to capture the OAuth 2.0 callback."""
    auth_code = None
    state_received = None

    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        CallbackHandler.auth_code = params.get("code", [None])[0]
        CallbackHandler.state_received = params.get("state", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        if CallbackHandler.auth_code:
            self.wfile.write(
                b"<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
                b"<h1>Authorization successful!</h1>"
                b"<p>You can close this tab and return to the terminal.</p>"
                b"</body></html>"
            )
        else:
            error = params.get("error", ["unknown"])[0]
            desc = params.get("error_description", [""])[0]
            msg = f"Authorization failed: {error} — {desc}".encode()
            self.wfile.write(
                b"<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
                b"<h1>" + msg + b"</h1></body></html>"
            )

    def log_message(self, format, *args):
        pass  # Suppress server logs


def exchange_code_for_token(code, code_verifier, client_id, client_secret):
    """Exchange authorization code for access + refresh tokens."""
    data = urllib.parse.urlencode({
        "code": code,
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": code_verifier,
    }).encode("utf-8")

    # Confidential client: HTTP Basic auth with client_id:client_secret
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    req = urllib.request.Request(
        TOKEN_URL,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {credentials}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main():
    cfg, config_path = load_config()
    x_config = cfg.get("services", {}).get("x", {})

    # Get client_id — from CLI arg, config, or prompt
    client_id = None

    # Check CLI args
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--client-id" and i < len(sys.argv) - 1:
            client_id = sys.argv[i + 1]
        elif arg.startswith("--client-id="):
            client_id = arg.split("=", 1)[1]

    if not client_id:
        client_id = x_config.get("oauth2_client_id", "")

    if not client_id:
        print("=" * 60)
        print("OAuth 2.0 Client ID Required")
        print("=" * 60)
        print()
        print("Go to: https://developer.x.com/en/portal/dashboard")
        print("  > Your App > Keys and tokens > OAuth 2.0 Client ID")
        print()
        client_id = input("Paste your Client ID here: ").strip()
        if not client_id:
            print("ERROR: Client ID is required.")
            sys.exit(1)

    client_secret = x_config.get("oauth2_client_secret", "")
    if not client_secret:
        print("ERROR: oauth2_client_secret not found in alfred.json services.x")
        sys.exit(1)

    # Save client_id to config for future use
    x_config["oauth2_client_id"] = client_id
    cfg["services"]["x"] = x_config
    save_config(cfg, config_path)

    # Generate PKCE
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(32)

    # Build authorization URL
    auth_params = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    })
    auth_url = f"{AUTH_URL}?{auth_params}"

    # Start callback server
    server = HTTPServer(("localhost", 3000), CallbackHandler)
    server_thread = threading.Thread(target=server.handle_request)
    server_thread.daemon = True
    server_thread.start()

    print()
    print("Opening browser for X authorization...")
    print(f"If it doesn't open automatically, visit:")
    print(f"  {auth_url}")
    print()
    webbrowser.open(auth_url)

    print("Waiting for authorization callback on http://localhost:3000 ...")
    server_thread.join(timeout=120)
    server.server_close()

    if not CallbackHandler.auth_code:
        print("ERROR: No authorization code received. Timed out or user denied.")
        sys.exit(1)

    if CallbackHandler.state_received != state:
        print("ERROR: State mismatch — possible CSRF attack. Aborting.")
        sys.exit(1)

    print("Authorization code received! Exchanging for tokens...")

    try:
        token_data = exchange_code_for_token(
            CallbackHandler.auth_code, code_verifier, client_id, client_secret
        )
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"ERROR: Token exchange failed — HTTP {e.code}")
        print(error_body)
        sys.exit(1)

    # Store tokens in alfred.json
    x_config["oauth2_access_token"] = token_data.get("access_token", "")
    x_config["oauth2_refresh_token"] = token_data.get("refresh_token", "")
    x_config["oauth2_token_type"] = token_data.get("token_type", "bearer")
    x_config["oauth2_scope"] = token_data.get("scope", "")
    x_config["oauth2_expires_at"] = int(time.time()) + token_data.get("expires_in", 7200)

    cfg["services"]["x"] = x_config
    save_config(cfg, config_path)

    print()
    print("=" * 60)
    print("SUCCESS — OAuth 2.0 tokens saved to alfred.json")
    print("=" * 60)
    print(f"  Access token:  {token_data.get('access_token', '')[:30]}...")
    print(f"  Refresh token: {'present' if token_data.get('refresh_token') else 'MISSING (no offline.access scope)'}")
    print(f"  Scopes:        {token_data.get('scope', '')}")
    print(f"  Expires in:    {token_data.get('expires_in', 0)} seconds")
    print()
    print("The X API tool will now use OAuth 2.0 for all requests.")
    print("Tokens auto-refresh when expired — no manual steps needed.")


if __name__ == "__main__":
    main()
