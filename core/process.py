"""Shared process management utilities — PID files, kill-and-wait, etc."""

import os
import signal
import time
from pathlib import Path
from typing import Optional


def read_pid(pid_file: Path) -> Optional[int]:
    """Read a PID from a file, returning None if missing/invalid."""
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def is_alive(pid: int) -> bool:
    """Check if a process is alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def kill_and_wait(pid: int, timeout: float = 5.0) -> bool:
    """Send SIGTERM, wait up to `timeout` seconds, then SIGKILL if needed.

    Returns True if process was stopped, False if it was already dead.
    """
    if not is_alive(pid):
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.1)
        if not is_alive(pid):
            return True

    # Still alive — force kill
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.3)
    except ProcessLookupError:
        pass
    return True


def cleanup_stale_pid(pid_file: Path) -> None:
    """Remove a PID file if the process it references is dead."""
    pid = read_pid(pid_file)
    if pid is not None and not is_alive(pid):
        pid_file.unlink(missing_ok=True)
