"""
Alfred AI — Persistent Metrics Storage

Time-series metrics backed by SQLite. One row per agent interaction,
aggregated on query with GROUP BY for day/week/month/year views.

Storage: data/metrics.db (~200 bytes/event, ~26 MB/year at 1000 msgs/day)
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import Config

DB_PATH = Config.DATA_DIR / "metrics.db"

# Time offsets for SQLite datetime() function
_PERIOD_OFFSETS = {
    "day": "-1 day",
    "week": "-7 days",
    "month": "-30 days",
    "year": "-365 days",
    "all": "-1000 years",
}

_AGENT_QUERY = """
SELECT
    agent,
    SUM(messages) as messages,
    SUM(tool_calls) as tool_calls,
    SUM(CASE WHEN is_error = 1 THEN 1 ELSE 0 END) as errors,
    SUM(elapsed_ms) as total_elapsed_ms,
    SUM(input_tokens) as input_tokens,
    SUM(output_tokens) as output_tokens,
    MAX(timestamp) as last_activity
FROM events
WHERE timestamp >= datetime('now', ?)
{agent_filter}
GROUP BY agent
ORDER BY agent
"""

_MODEL_QUERY = """
SELECT
    provider || '/' || model as model_key,
    SUM(messages) as messages,
    SUM(tool_calls) as tool_calls,
    SUM(CASE WHEN is_error = 1 THEN 1 ELSE 0 END) as errors,
    SUM(elapsed_ms) as total_elapsed_ms,
    SUM(input_tokens) as input_tokens,
    SUM(output_tokens) as output_tokens,
    MAX(timestamp) as last_activity
FROM events
WHERE timestamp >= datetime('now', ?)
  AND provider != '' AND model != ''
{agent_filter}
GROUP BY provider, model
ORDER BY model_key
"""


class MetricsStore:
    """
    Time-series metrics storage backed by SQLite.

    Write one row per agent interaction. Query with aggregation
    for day/week/month/year views. Returns same shape as
    AgentMetrics.snapshot() for seamless API compatibility.
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """Get a connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    agent         TEXT    NOT NULL,
                    provider      TEXT    DEFAULT '',
                    model         TEXT    DEFAULT '',
                    messages      INTEGER DEFAULT 1,
                    tool_calls    INTEGER DEFAULT 0,
                    elapsed_ms    INTEGER DEFAULT 0,
                    input_tokens  INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    is_error      INTEGER DEFAULT 0,
                    error_text    TEXT    DEFAULT ''
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent, timestamp)"
            )

    def record(self, agent: str, provider: str, model: str,
               elapsed_ms: int, tool_calls: int,
               input_tokens: int, output_tokens: int):
        """Record a successful agent interaction."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO events
                       (timestamp, agent, provider, model, messages, tool_calls,
                        elapsed_ms, input_tokens, output_tokens, is_error)
                       VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, 0)""",
                    (now, agent, provider or "", model or "",
                     tool_calls, elapsed_ms, input_tokens, output_tokens),
                )
        except Exception:
            pass  # Never let metrics storage break the agent

    def record_error(self, agent: str, provider: str, model: str, error: str):
        """Record an agent error event."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO events
                       (timestamp, agent, provider, model, messages, tool_calls,
                        elapsed_ms, input_tokens, output_tokens, is_error, error_text)
                       VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, 1, ?)""",
                    (now, agent, provider or "", model or "", error[:500]),
                )
        except Exception:
            pass

    def query(self, period: str = "day", agent: str = None) -> dict:
        """
        Aggregate metrics for a time period.

        Args:
            period: day|week|month|year|all
            agent: Optional agent name filter

        Returns:
            Same shape as AgentMetrics.snapshot():
            {
                "timestamp": "...",
                "period": "day",
                "agents": { "alfred": { messages, tokens, ... } },
                "models": { "anthropic/claude-...": { ... } }
            }
        """
        offset = _PERIOD_OFFSETS.get(period, _PERIOD_OFFSETS["day"])
        agent_filter = "AND agent = ?" if agent else ""

        params_base = [offset]
        if agent:
            params_base.append(agent)

        agents_result = {}
        models_result = {}

        try:
            with self._conn() as conn:
                # Agent aggregation
                sql = _AGENT_QUERY.format(agent_filter=agent_filter)
                for row in conn.execute(sql, params_base):
                    total_ms = row["total_elapsed_ms"] or 0
                    msgs = row["messages"] or 0
                    agents_result[row["agent"]] = {
                        "messages": msgs,
                        "tool_calls": row["tool_calls"] or 0,
                        "errors": row["errors"] or 0,
                        "total_elapsed_ms": total_ms,
                        "input_tokens": row["input_tokens"] or 0,
                        "output_tokens": row["output_tokens"] or 0,
                        "last_activity": row["last_activity"],
                        "avg_ms": total_ms // msgs if msgs else 0,
                        "total_tokens": (row["input_tokens"] or 0) + (row["output_tokens"] or 0),
                        "recent_errors": [],
                    }

                # Model aggregation
                sql = _MODEL_QUERY.format(agent_filter=agent_filter)
                for row in conn.execute(sql, params_base):
                    total_ms = row["total_elapsed_ms"] or 0
                    msgs = row["messages"] or 0
                    models_result[row["model_key"]] = {
                        "messages": msgs,
                        "tool_calls": row["tool_calls"] or 0,
                        "errors": row["errors"] or 0,
                        "total_elapsed_ms": total_ms,
                        "input_tokens": row["input_tokens"] or 0,
                        "output_tokens": row["output_tokens"] or 0,
                        "last_activity": row["last_activity"],
                        "avg_ms": total_ms // msgs if msgs else 0,
                        "total_tokens": (row["input_tokens"] or 0) + (row["output_tokens"] or 0),
                    }
        except Exception:
            pass  # Return empty on error

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "period": period,
            "agents": agents_result,
            "models": models_result,
        }

    def totals(self) -> dict:
        """
        Get all-time totals per agent and per model.

        Used to seed in-memory counters on startup so the session view
        starts from historical totals instead of zero.
        """
        return self.query(period="all")
