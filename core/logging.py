"""
Alfred AI - Structured Logging + Agent Metrics

Provides a unified logging setup for all Alfred components.
All logs go to data/alfred.log (when running as daemon) and/or console.

Usage:
    from core.logging import get_logger, metrics

    logger = get_logger("agent")       # -> "alfred.agent"
    logger.info("Processing message")
    logger.warning("Something off")

    # Track agent metrics
    metrics.record_message("alfred", elapsed_ms=350, tool_calls=2)
    metrics.record_error("alfred", "LLM timeout")
    print(metrics.summary("alfred"))
"""

import logging
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional

from .config import config

# ─── Log File ──────────────────────────────────────────────────

LOG_DIR = config.PROJECT_ROOT / "data"
LOG_FILE = LOG_DIR / "alfred.log"

# ─── Logger Factory ───────────────────────────────────────────

_initialized = False


def _ensure_log_dir():
    """Create log directory if needed."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO, to_file: bool = False,
                  to_console: bool = True):
    """
    Initialize Alfred's logging system.

    Call this once at startup (e.g., in cmd_start or cmd_api_start).
    Subsequent calls are no-ops.

    Args:
        level: Log level (default INFO)
        to_file: Also write to data/alfred.log
        to_console: Also write to console (default True)
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger("alfred")
    root.setLevel(level)

    # Avoid duplicate handlers
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (clean format)
    if to_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console_fmt = logging.Formatter("  [%(name)s] %(message)s")
        console.setFormatter(console_fmt)
        root.addHandler(console)

    # File handler (full timestamps)
    if to_file:
        _ensure_log_dir()
        fh = logging.FileHandler(str(LOG_FILE), mode="a")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for name in (
        "httpx", "httpcore", "urllib3",
        "sentence_transformers", "huggingface_hub", "transformers",
        "discord", "discord.gateway", "discord.client", "discord.http",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a namespaced logger.

    Args:
        name: Component name (e.g., "agent", "memory", "discord")

    Returns:
        Logger named "alfred.<name>"
    """
    # Auto-initialize with console-only if not yet set up
    if not _initialized:
        setup_logging(to_console=True, to_file=False)
    return logging.getLogger(f"alfred.{name}")


# ─── Agent Metrics ─────────────────────────────────────────────

class AgentMetrics:
    """
    In-memory metrics tracker for agent activity.

    Tracks per-agent:
    - Total messages processed
    - Total tool calls
    - Total errors
    - Average response time
    - Last activity timestamp
    - Recent error log (last 10)

    Metrics are session-scoped (reset on restart). For persistent
    monitoring, use the metrics_snapshot() method to save to disk.
    """

    def __init__(self):
        self._data: dict[str, dict] = defaultdict(lambda: {
            "messages": 0,
            "tool_calls": 0,
            "errors": 0,
            "total_elapsed_ms": 0,
            "last_activity": None,
            "recent_errors": [],  # Last 10 errors
        })

    def record_message(self, agent_name: str, elapsed_ms: int = 0,
                       tool_calls: int = 0):
        """Record a successful agent interaction."""
        d = self._data[agent_name]
        d["messages"] += 1
        d["tool_calls"] += tool_calls
        d["total_elapsed_ms"] += elapsed_ms
        d["last_activity"] = datetime.now(timezone.utc).isoformat()

    def record_error(self, agent_name: str, error: str):
        """Record an agent error."""
        d = self._data[agent_name]
        d["errors"] += 1
        d["last_activity"] = datetime.now(timezone.utc).isoformat()
        d["recent_errors"].append({
            "error": error[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Keep only last 10
        d["recent_errors"] = d["recent_errors"][-10:]

    def summary(self, agent_name: str = None) -> dict:
        """
        Get metrics summary.

        Args:
            agent_name: Specific agent, or None for all agents.

        Returns:
            Dict with metrics. Includes avg_ms if messages > 0.
        """
        if agent_name:
            d = self._data[agent_name]
            result = dict(d)
            if d["messages"] > 0:
                result["avg_ms"] = d["total_elapsed_ms"] // d["messages"]
            else:
                result["avg_ms"] = 0
            return result

        # All agents
        return {
            name: self.summary(name)
            for name in sorted(self._data.keys())
        }

    def snapshot(self) -> dict:
        """Get a JSON-serializable snapshot of all metrics."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents": self.summary(),
        }

    def save_snapshot(self, path: Path = None):
        """Save metrics snapshot to disk."""
        path = path or (LOG_DIR / "metrics.json")
        _ensure_log_dir()
        path.write_text(json.dumps(self.snapshot(), indent=2, default=str))


# Singleton metrics instance
metrics = AgentMetrics()
