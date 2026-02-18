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
            "input_tokens": 0,
            "output_tokens": 0,
            "last_activity": None,
            "recent_errors": [],  # Last 10 errors
        })
        # Per-model rollup — keyed by "provider/model"
        self._by_model: dict[str, dict] = defaultdict(lambda: {
            "messages": 0,
            "tool_calls": 0,
            "errors": 0,
            "total_elapsed_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "last_activity": None,
        })

        # Persistent storage — writes every event to SQLite
        try:
            from .metrics_store import MetricsStore
            self._store = MetricsStore()
        except Exception:
            self._store = None

    def record_message(self, agent_name: str, elapsed_ms: int = 0,
                       tool_calls: int = 0,
                       input_tokens: int = 0, output_tokens: int = 0,
                       provider: str = "", model: str = ""):
        """Record a successful agent interaction."""
        now = datetime.now(timezone.utc).isoformat()

        # Per-agent
        d = self._data[agent_name]
        d["messages"] += 1
        d["tool_calls"] += tool_calls
        d["total_elapsed_ms"] += elapsed_ms
        d["input_tokens"] += input_tokens
        d["output_tokens"] += output_tokens
        d["last_activity"] = now

        # Per-model
        if provider and model:
            model_key = f"{provider}/{model}"
            m = self._by_model[model_key]
            m["messages"] += 1
            m["tool_calls"] += tool_calls
            m["total_elapsed_ms"] += elapsed_ms
            m["input_tokens"] += input_tokens
            m["output_tokens"] += output_tokens
            m["last_activity"] = now

        # Persist to SQLite
        if self._store:
            self._store.record(agent_name, provider, model,
                               elapsed_ms, tool_calls,
                               input_tokens, output_tokens)

    def record_error(self, agent_name: str, error: str,
                     provider: str = "", model: str = ""):
        """Record an agent error."""
        now = datetime.now(timezone.utc).isoformat()

        d = self._data[agent_name]
        d["errors"] += 1
        d["last_activity"] = now
        d["recent_errors"].append({
            "error": error[:500],
            "timestamp": now,
        })
        # Keep only last 10
        d["recent_errors"] = d["recent_errors"][-10:]

        # Per-model
        if provider and model:
            model_key = f"{provider}/{model}"
            self._by_model[model_key]["errors"] += 1
            self._by_model[model_key]["last_activity"] = now

        # Persist to SQLite
        if self._store:
            self._store.record_error(agent_name, provider, model, error)

    def summary(self, agent_name: str = None) -> dict:
        """
        Get metrics summary.

        Args:
            agent_name: Specific agent, or None for all agents.

        Returns:
            Dict with metrics. Includes avg_ms and total_tokens if messages > 0.
        """
        if agent_name:
            d = self._data[agent_name]
            result = dict(d)
            if d["messages"] > 0:
                result["avg_ms"] = d["total_elapsed_ms"] // d["messages"]
            else:
                result["avg_ms"] = 0
            result["total_tokens"] = d["input_tokens"] + d["output_tokens"]
            return result

        # All agents
        return {
            name: self.summary(name)
            for name in sorted(self._data.keys())
        }

    def model_summary(self) -> dict:
        """
        Get per-model metrics summary.

        Returns:
            Dict keyed by "provider/model" with same shape as agent metrics.
        """
        result = {}
        for model_key in sorted(self._by_model.keys()):
            m = self._by_model[model_key]
            entry = dict(m)
            if m["messages"] > 0:
                entry["avg_ms"] = m["total_elapsed_ms"] // m["messages"]
            else:
                entry["avg_ms"] = 0
            entry["total_tokens"] = m["input_tokens"] + m["output_tokens"]
            result[model_key] = entry
        return result

    def snapshot(self) -> dict:
        """Get a JSON-serializable snapshot of all metrics."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents": self.summary(),
            "models": self.model_summary(),
        }

    def save_snapshot(self, path: Path = None):
        """Save metrics snapshot to disk."""
        path = path or (LOG_DIR / "metrics.json")
        _ensure_log_dir()
        path.write_text(json.dumps(self.snapshot(), indent=2, default=str))


# Singleton metrics instance
metrics = AgentMetrics()
