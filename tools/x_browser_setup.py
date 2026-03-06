#!/usr/bin/env python3
"""
X Browser Login Setup — Persistent Brave Profile

Opens Brave with a dedicated profile. Log in manually.
The profile persists, so ALFRED agents stay logged in for future sessions.

Usage:
    cd ~/.alfred-ai && .venv/bin/python tools/x_browser_setup.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from playwright.sync_api import sync_playwright
from core.config import config

PROFILE_DIR = Path(config.DATA_DIR) / ".browser_state" / "brave_profile"
STATE_FILE = Path(config.DATA_DIR) / ".browser_state" / "x_session.json"


def main():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 50)
    print("  X Browser Login Setup (Brave)")
    print("=" * 50)
    print()
    print("A Brave window will open to x.com/login.")
    print("Log into @DarkStoneCap manually.")
    print("Once you see the home timeline, come back")
    print("here and press Enter.")
    print()

    pw = sync_playwright().start()

    # Persistent context = real browser profile on disk
    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        executable_path="/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        headless=False,
        viewport={"width": 1280, "height": 900},
        locale="en-US",
        timezone_id="America/New_York",
        args=["--disable-blink-features=AutomationControlled"],
    )

    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://x.com/i/flow/login", wait_until="domcontentloaded", timeout=30000)

    print("Brave opened. Log in now...")
    print()
    input(">>> Press Enter after you've logged in and see the home timeline... ")
    print()

    # Also save as Playwright storage_state for compatibility
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    context.storage_state(path=str(STATE_FILE))
    print(f"Session cookies saved to: {STATE_FILE}")
    print(f"Browser profile saved to: {PROFILE_DIR}")

    # Verify
    page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=15000)
    time.sleep(2)

    if "/login" not in page.url:
        print(f"Verified: logged in at {page.url}")
        print()
        print("ALFRED agents can now use x_browser_reply")
        print("and x_browser_quote to interact with tweets.")
    else:
        print("WARNING: Login may not have worked.")

    context.close()
    pw.stop()
    print("\nDone!")


if __name__ == "__main__":
    main()
