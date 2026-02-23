"""
Alfred AI - Scheduler
Cron-style task scheduling for agents.

Schedules are stored in alfred.json under each agent's config.
The scheduler runs as a background loop that checks what's due.

Schedule format (in alfred.json):
    "agents": {
        "trader": {
            "schedules": [
                {
                    "id": "morning-scan",
                    "cron": "30 9 * * 1-5",
                    "task": "Run morning market scan for TSLA, NVDA, SPY",
                    "enabled": true
                }
            ]
        }
    }

Cron format: minute hour day_of_month month day_of_week
    - Standard cron syntax
    - day_of_week: 0=Sun, 1=Mon, ..., 6=Sat (or 1-5 for weekdays)
    - Supports: *, specific values, ranges (1-5), lists (1,3,5), steps (*/5)
"""

import json
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .logging import get_logger

logger = get_logger("scheduler")
from dataclasses import dataclass, field, asdict

from .config import config, _load_config, _save_config


@dataclass
class ScheduleRun:
    """A single execution record for a scheduled task."""
    timestamp: str  # ISO timestamp
    result: str  # "success" or error message
    elapsed_ms: int = 0  # How long the run took
    is_catchup: bool = False  # Was this a missed-run catchup?
    is_retry: bool = False  # Was this a retry attempt?

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduleRun":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Schedule:
    """A scheduled task for an agent."""
    id: str
    cron: str  # "30 9 * * 1-5"
    task: str  # The message/instruction to send to the agent
    enabled: bool = True
    last_run: str = ""  # ISO timestamp of last execution
    last_result: str = ""  # "success" or error message
    created_at: str = ""

    # Retry settings
    max_retries: int = 0  # 0 = no retry, 1-3 = retry on failure
    retry_delay_seconds: int = 30  # Wait between retries

    # Run statistics
    run_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    consecutive_failures: int = 0

    # Run history (last N runs, newest first)
    history: list = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(config.tz).isoformat()
        # Deserialize history dicts into ScheduleRun objects
        if self.history and isinstance(self.history[0], dict):
            self.history = [ScheduleRun.from_dict(h) for h in self.history]

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure history is serializable
        d["history"] = [h if isinstance(h, dict) else asdict(h) for h in (self.history or [])]
        return d

    def record_run(self, result: str, elapsed_ms: int = 0, is_catchup: bool = False, is_retry: bool = False):
        """Record a run in history and update stats."""
        run = ScheduleRun(
            timestamp=datetime.now(config.tz).isoformat(),
            result=result,
            elapsed_ms=elapsed_ms,
            is_catchup=is_catchup,
            is_retry=is_retry,
        )
        self.history.insert(0, run)
        # Keep only the last 20 runs
        self.history = self.history[:20]

        self.last_run = run.timestamp
        self.last_result = result
        self.run_count += 1

        if result == "success":
            self.success_count += 1
            self.consecutive_failures = 0
        else:
            self.fail_count += 1
            self.consecutive_failures += 1

            # Auto-disable after 5 consecutive failures
            if self.consecutive_failures >= 5 and self.enabled:
                self.enabled = False
                logger.warning(
                    f"Schedule '{self.id}' auto-disabled after "
                    f"{self.consecutive_failures} consecutive failures"
                )
                try:
                    from .alerting import send_schedule_failure_alert
                    send_schedule_failure_alert(
                        self.id, self.consecutive_failures, result
                    )
                except Exception:
                    pass  # Never let alerting break the scheduler

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage (0-100)."""
        if self.run_count == 0:
            return 0.0
        return (self.success_count / self.run_count) * 100

    @classmethod
    def from_dict(cls, data: dict) -> "Schedule":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def parse_cron_field(field_str: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of matching values."""
    values = set()

    for part in field_str.split(","):
        part = part.strip()

        if part == "*":
            values.update(range(min_val, max_val + 1))
        elif "/" in part:
            # Step: */5 or 10-30/5
            base, step = part.split("/", 1)
            step = int(step)
            if base == "*":
                start = min_val
                end = max_val
            elif "-" in base:
                start, end = base.split("-", 1)
                start, end = int(start), int(end)
            else:
                start = int(base)
                end = max_val
            values.update(range(start, end + 1, step))
        elif "-" in part:
            # Range: 1-5
            start, end = part.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            # Single value
            values.add(int(part))

    return values


def cron_matches(cron_str: str, dt: datetime = None) -> bool:
    """Check if a cron expression matches the given datetime (default: now)."""
    if dt is None:
        dt = datetime.now(config.tz)

    parts = cron_str.strip().split()
    if len(parts) != 5:
        return False

    minute_str, hour_str, dom_str, month_str, dow_str = parts

    try:
        minutes = parse_cron_field(minute_str, 0, 59)
        hours = parse_cron_field(hour_str, 0, 23)
        doms = parse_cron_field(dom_str, 1, 31)
        months = parse_cron_field(month_str, 1, 12)
        dows = parse_cron_field(dow_str, 0, 6)  # 0=Sun
    except (ValueError, IndexError):
        return False

    # Python weekday: 0=Mon, 6=Sun. Cron: 0=Sun, 6=Sat
    cron_dow = (dt.weekday() + 1) % 7  # Convert to cron format

    return (
        dt.minute in minutes
        and dt.hour in hours
        and dt.day in doms
        and dt.month in months
        and cron_dow in dows
    )


def next_run(cron_str: str, after: datetime = None, max_search_days: int = 7) -> Optional[datetime]:
    """
    Calculate the next time a cron expression will match.

    Scans forward minute-by-minute from `after` (default: now).
    Returns None if no match found within max_search_days.
    """
    if after is None:
        after = datetime.now(config.tz)

    # Start from the next minute
    check = after.replace(second=0, microsecond=0)
    from datetime import timedelta
    check += timedelta(minutes=1)

    end = after + timedelta(days=max_search_days)

    while check <= end:
        if cron_matches(cron_str, check):
            return check
        check += timedelta(minutes=1)

    return None


def describe_cron(cron_str: str) -> str:
    """Human-readable description of a cron expression."""
    parts = cron_str.strip().split()
    if len(parts) != 5:
        return cron_str

    minute, hour, dom, month, dow = parts

    # Common patterns
    dow_names = {
        "1-5": "weekdays",
        "0,6": "weekends",
        "*": "every day",
        "1": "Mon", "2": "Tue", "3": "Wed", "4": "Thu", "5": "Fri",
        "6": "Sat", "0": "Sun",
    }

    time_str = ""
    if minute != "*" and hour != "*":
        # Handle hour ranges like "10-15"
        if "-" in hour and not hour.startswith("*/"):
            h_start, h_end = hour.split("-", 1)
            time_str = f":{int(minute):02d} {h_start}-{h_end}h"
        elif hour.isdigit() and minute.isdigit():
            time_str = f"{int(hour):02d}:{int(minute):02d}"
        else:
            time_str = f"{hour}:{minute}"
    elif minute.startswith("*/"):
        time_str = f"every {minute[2:]} min"
    elif hour.startswith("*/"):
        time_str = f"every {hour[2:]} hr"
    elif minute != "*" and hour == "*":
        time_str = f":{int(minute):02d} every hour" if minute.isdigit() else f":{minute} every hour"

    dow_desc = dow_names.get(dow, dow)

    if dom == "*" and month == "*":
        return f"{time_str} {dow_desc}".strip()
    return cron_str


# ─── Schedule Management ────────────────────────────────────────

def get_agent_schedules(agent_name: str) -> list[Schedule]:
    """Get all schedules for an agent."""
    cfg = _load_config()
    agent_cfg = cfg.get("agents", {}).get(agent_name, {})
    schedules_data = agent_cfg.get("schedules", [])
    return [Schedule.from_dict(s) for s in schedules_data]


def add_schedule(agent_name: str, cron: str, task: str) -> Schedule:
    """Add a new schedule to an agent."""
    cfg = _load_config()

    if agent_name not in cfg.get("agents", {}):
        raise ValueError(f"Agent '{agent_name}' not found")

    schedule = Schedule(
        id=str(uuid.uuid4())[:8],
        cron=cron,
        task=task,
    )

    if "schedules" not in cfg["agents"][agent_name]:
        cfg["agents"][agent_name]["schedules"] = []

    cfg["agents"][agent_name]["schedules"].append(schedule.to_dict())
    _save_config(cfg)

    return schedule


def remove_schedule(agent_name: str, schedule_id: str) -> bool:
    """Remove a schedule from an agent."""
    cfg = _load_config()

    if agent_name not in cfg.get("agents", {}):
        return False

    schedules = cfg["agents"][agent_name].get("schedules", [])
    original_len = len(schedules)
    cfg["agents"][agent_name]["schedules"] = [
        s for s in schedules if s.get("id") != schedule_id
    ]

    if len(cfg["agents"][agent_name]["schedules"]) < original_len:
        _save_config(cfg)
        return True
    return False


def toggle_schedule(agent_name: str, schedule_id: str, enabled: bool) -> bool:
    """Enable or disable a specific schedule."""
    cfg = _load_config()

    if agent_name not in cfg.get("agents", {}):
        return False

    for s in cfg["agents"][agent_name].get("schedules", []):
        if s.get("id") == schedule_id:
            s["enabled"] = enabled
            _save_config(cfg)
            return True
    return False


def update_schedule_result(agent_name: str, schedule_id: str, result: str,
                           elapsed_ms: int = 0, is_catchup: bool = False, is_retry: bool = False):
    """Record a run result with full history tracking."""
    cfg = _load_config()

    if agent_name not in cfg.get("agents", {}):
        return

    for s in cfg["agents"][agent_name].get("schedules", []):
        if s.get("id") == schedule_id:
            schedule = Schedule.from_dict(s)
            schedule.record_run(
                result=result[:200],
                elapsed_ms=elapsed_ms,
                is_catchup=is_catchup,
                is_retry=is_retry,
            )
            # Write the updated schedule back
            updated = schedule.to_dict()
            s.update(updated)
            _save_config(cfg)
            return


# ─── Scheduler Loop ─────────────────────────────────────────────

# Max age for missed-run catchup (don't run tasks missed more than 1 hour ago)
MAX_MISSED_RUN_MINUTES = 60


class Scheduler:
    """
    Background scheduler that checks for due tasks every minute.

    Features:
    - Missed-run detection: catches up on tasks that were due while Alfred was offline
    - Non-blocking execution: each task runs in its own thread
    - Error isolation: one failing task doesn't block others
    - Dedup: never runs the same schedule twice in the same minute

    Usage:
        scheduler = Scheduler()
        scheduler.start()  # Runs in background thread
        # ... later ...
        scheduler.stop()
    """

    def __init__(self, agent_runner=None):
        """
        Args:
            agent_runner: Callable(agent_name, task_message) -> str
                          Function that runs a task on an agent and returns the result.
        """
        self._runner = agent_runner
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_check_minute = -1
        self._active_tasks: dict[str, threading.Thread] = {}  # schedule_id -> thread

    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            return
        self._running = True

        # Check for missed runs on startup before entering the main loop
        self._check_missed_runs()

        self._thread = threading.Thread(target=self._loop, daemon=True, name="scheduler")
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        # Wait for active tasks to finish (with timeout)
        for sid, t in list(self._active_tasks.items()):
            t.join(timeout=10)
        self._active_tasks.clear()
        logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def active_task_count(self) -> int:
        """Number of currently executing scheduled tasks."""
        # Clean up finished threads
        self._active_tasks = {
            sid: t for sid, t in self._active_tasks.items() if t.is_alive()
        }
        return len(self._active_tasks)

    def trigger_now(self, agent_name: str, schedule_id: str) -> dict:
        """Manually trigger a schedule to run immediately (non-blocking).

        Returns a dict with status: started | already_running | not_found | no_runner
        """
        schedule = get_schedule(agent_name, schedule_id)
        if not schedule:
            return {"status": "not_found", "agent": agent_name, "schedule_id": schedule_id}

        if not self._runner:
            return {"status": "no_runner", "agent": agent_name, "schedule_id": schedule_id}

        # Clean up finished threads
        self._active_tasks = {
            sid: t for sid, t in self._active_tasks.items() if t.is_alive()
        }

        if schedule_id in self._active_tasks:
            return {"status": "already_running", "agent": agent_name, "schedule_id": schedule_id}

        # Reuse the existing _execute_schedule — same threading, retries, result recording
        self._execute_schedule(agent_name, schedule, is_catchup=False)

        return {"status": "started", "agent": agent_name, "schedule_id": schedule_id}

    def is_schedule_running(self, schedule_id: str) -> bool:
        """Check if a specific schedule is currently executing."""
        thread = self._active_tasks.get(schedule_id)
        return thread is not None and thread.is_alive()

    def _loop(self):
        """Main scheduler loop — checks every 30 seconds."""
        while self._running:
            try:
                now = datetime.now(config.tz)

                # Only check once per minute
                current_minute = now.hour * 60 + now.minute
                if current_minute != self._last_check_minute:
                    self._last_check_minute = current_minute
                    self._check_schedules(now)
                    # System maintenance (memory compaction, etc.)
                    self._check_maintenance(now)
            except Exception as e:
                # Never let the scheduler loop die
                logger.error(f"Scheduler loop error: {e}")

            # Sleep in small increments so stop() is responsive
            for _ in range(6):
                if not self._running:
                    break
                time.sleep(5)

    def _check_missed_runs(self):
        """
        On startup, check if any tasks were missed while Alfred was offline.

        A run is considered "missed" if:
        1. The cron would have matched at some point since last_run
        2. last_run was more than 1 minute ago but less than MAX_MISSED_RUN_MINUTES ago
        3. The schedule is enabled and the agent is active

        Only runs the task once (not for every missed minute).
        """
        cfg = _load_config()
        now = datetime.now(config.tz)
        caught_up = 0

        for agent_name, agent_cfg in cfg.get("agents", {}).items():
            if agent_cfg.get("status") == "paused":
                continue

            for schedule_data in agent_cfg.get("schedules", []):
                schedule = Schedule.from_dict(schedule_data)
                if not schedule.enabled:
                    continue

                if not schedule.last_run:
                    continue  # Never ran — don't assume it was "missed"

                try:
                    last = datetime.fromisoformat(schedule.last_run)
                    # Existing timestamps may be naive — assume configured timezone
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=config.tz)
                except (ValueError, TypeError):
                    continue

                minutes_since = (now - last).total_seconds() / 60

                # Skip if last run was recent (less than 2 min ago) or too old
                if minutes_since < 2 or minutes_since > MAX_MISSED_RUN_MINUTES:
                    continue

                # Check if the cron would have matched at any minute since last_run
                # (scan in 1-minute increments, stop at first match)
                check_time = last.replace(second=0, microsecond=0)
                missed = False
                from datetime import timedelta as _td
                for _ in range(int(min(minutes_since, MAX_MISSED_RUN_MINUTES))):
                    check_time = check_time + _td(minutes=1)
                    if cron_matches(schedule.cron, check_time):
                        missed = True
                        break

                if missed:
                    logger.info(
                        f"Missed run detected: '{schedule.id}' for {agent_name} "
                        f"(last ran {int(minutes_since)}m ago)"
                    )
                    self._execute_schedule(agent_name, schedule, is_catchup=True)
                    caught_up += 1

        if caught_up:
            logger.info(f"Caught up on {caught_up} missed task(s)")

    def _check_schedules(self, now: datetime):
        """Check all agent schedules and run any that are due."""
        try:
            cfg = _load_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return

        for agent_name, agent_cfg in cfg.get("agents", {}).items():
            # Skip paused agents
            if agent_cfg.get("status") == "paused":
                continue

            for schedule_data in agent_cfg.get("schedules", []):
                try:
                    schedule = Schedule.from_dict(schedule_data)
                except Exception:
                    continue

                if not schedule.enabled:
                    continue

                if cron_matches(schedule.cron, now):
                    # Check if we already ran this minute
                    if schedule.last_run:
                        try:
                            last = datetime.fromisoformat(schedule.last_run)
                            if last.tzinfo is None:
                                last = last.replace(tzinfo=config.tz)
                            if (last.hour == now.hour
                                    and last.minute == now.minute
                                    and last.date() == now.date()):
                                continue  # Already ran this minute
                        except (ValueError, TypeError):
                            pass

                    # Don't start if this schedule is already running
                    if schedule.id in self._active_tasks:
                        if self._active_tasks[schedule.id].is_alive():
                            logger.warning(
                                f"'{schedule.id}' still running from previous trigger, skipping"
                            )
                            continue

                    self._execute_schedule(agent_name, schedule)

    _last_maintenance_day: str = ""

    def _check_maintenance(self, now: datetime):
        """
        Run system maintenance tasks (memory compaction, etc.).

        Checks alfred.json for maintenance config:
        {
            "maintenance": {
                "memory_compact_cron": "0 3 * * 0",  // Sunday 3 AM
                "compact_max_age_days": 90,
                "compact_min_importance": 0.3
            }
        }

        Falls back to weekly Sunday 3 AM if not configured.
        """
        cfg = _load_config()
        maintenance = cfg.get("maintenance", {})
        compact_cron = maintenance.get("memory_compact_cron", "0 3 * * 0")  # Default: Sunday 3 AM

        if not cron_matches(compact_cron, now):
            return

        # Only run once per day (avoid re-running if loop checks the same minute twice)
        today_key = now.strftime("%Y-%m-%d")
        if self._last_maintenance_day == today_key:
            return
        self._last_maintenance_day = today_key

        # Run compaction in a background thread
        max_age = maintenance.get("compact_max_age_days", 90)
        min_importance = maintenance.get("compact_min_importance", 0.3)

        def _run_compact():
            try:
                from .memory import MemoryStore
                store = MemoryStore()
                results = store.compact(max_age_days=max_age, min_importance=min_importance)
                total = sum(results.values())
                if total > 0:
                    logger.info(f"Maintenance: compacted {total} old low-importance memories")
                else:
                    logger.debug("Maintenance: no memories needed compaction")
            except Exception as e:
                logger.warning(f"Maintenance compaction failed: {e}")

            # Session cleanup — remove session files older than 30 days
            try:
                session_max_age = maintenance.get("session_max_age_days", 30)
                from datetime import timedelta
                cutoff = datetime.now() - timedelta(days=session_max_age)
                cleanup_count = 0
                workspace_root = config.PROJECT_ROOT / "workspaces"
                if workspace_root.exists():
                    for session_file in workspace_root.rglob("session_*.json"):
                        if session_file.stat().st_mtime < cutoff.timestamp():
                            session_file.unlink()
                            cleanup_count += 1
                if cleanup_count > 0:
                    logger.info(f"Maintenance: cleaned up {cleanup_count} old session files")
            except Exception as e:
                logger.warning(f"Session cleanup failed: {e}")

            # Metrics cleanup — remove events older than 90 days
            try:
                metrics_max_age = maintenance.get("metrics_max_age_days", 90)
                import sqlite3
                db_path = config.DATA_DIR / "metrics.db"
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    from datetime import datetime as _dt, timedelta as _td
                    cutoff = (_dt.now(config.tz) - _td(days=metrics_max_age)).strftime("%Y-%m-%d %H:%M:%S")
                    cursor = conn.execute(
                        "DELETE FROM events WHERE timestamp < ?",
                        (cutoff,),
                    )
                    deleted = cursor.rowcount
                    conn.commit()
                    conn.close()
                    if deleted > 0:
                        logger.info(f"Maintenance: pruned {deleted} old metric events")
            except Exception as e:
                logger.warning(f"Metrics cleanup failed: {e}")

        t = threading.Thread(target=_run_compact, daemon=True, name="maintenance-compact")
        t.start()

    def _execute_schedule(self, agent_name: str, schedule: Schedule, is_catchup: bool = False):
        """Execute a scheduled task in a separate thread with retry support."""
        tag = " (catchup)" if is_catchup else ""
        logger.info(f"Running '{schedule.id}' for {agent_name}{tag}: {schedule.task[:60]}")

        if not self._runner:
            update_schedule_result(agent_name, schedule.id, "no runner configured")
            logger.warning("No agent runner configured, skipping execution")
            return

        def _run():
            max_attempts = 1 + max(0, min(schedule.max_retries, 3))  # Cap at 3 retries
            retry_delay = max(5, schedule.retry_delay_seconds)  # Min 5s between retries

            for attempt in range(max_attempts):
                is_retry = attempt > 0
                if is_retry:
                    logger.info(f"'{schedule.id}' retry {attempt}/{schedule.max_retries}")
                    time.sleep(retry_delay)
                    # Check if scheduler was stopped during the wait
                    if not self._running:
                        return

                start_time = time.monotonic()
                try:
                    self._runner(agent_name, schedule.task)
                    elapsed = int((time.monotonic() - start_time) * 1000)
                    update_schedule_result(
                        agent_name, schedule.id, "success",
                        elapsed_ms=elapsed, is_catchup=is_catchup, is_retry=is_retry,
                    )
                    logger.info(f"'{schedule.id}' completed in {elapsed}ms")
                    break  # Success — done

                except Exception as e:
                    elapsed = int((time.monotonic() - start_time) * 1000)
                    is_last_attempt = attempt == max_attempts - 1
                    result = f"error: {e}"

                    if not is_last_attempt:
                        result = f"error (will retry): {e}"
                        logger.warning(f"'{schedule.id}' failed (attempt {attempt + 1}/{max_attempts}): {e}")
                    else:
                        logger.error(f"'{schedule.id}' failed: {e}")

                    update_schedule_result(
                        agent_name, schedule.id, result,
                        elapsed_ms=elapsed, is_catchup=is_catchup, is_retry=is_retry,
                    )

            # Clean up from active tasks when done (success or all retries exhausted)
            self._active_tasks.pop(schedule.id, None)

        task_thread = threading.Thread(
            target=_run, daemon=True,
            name=f"schedule-{schedule.id}",
        )
        self._active_tasks[schedule.id] = task_thread
        task_thread.start()


# ─── Convenience ─────────────────────────────────────────────────

def get_all_schedules() -> dict[str, list[Schedule]]:
    """Get all schedules for all agents."""
    cfg = _load_config()
    result = {}
    for agent_name, agent_cfg in cfg.get("agents", {}).items():
        schedules = [Schedule.from_dict(s) for s in agent_cfg.get("schedules", [])]
        if schedules:
            result[agent_name] = schedules
    return result


def get_schedule(agent_name: str, schedule_id: str) -> Optional[Schedule]:
    """Get a specific schedule by ID."""
    cfg = _load_config()
    agent_cfg = cfg.get("agents", {}).get(agent_name, {})
    for s in agent_cfg.get("schedules", []):
        if s.get("id") == schedule_id:
            return Schedule.from_dict(s)
    return None


def get_schedule_history(agent_name: str, schedule_id: str) -> list[ScheduleRun]:
    """Get run history for a specific schedule."""
    schedule = get_schedule(agent_name, schedule_id)
    if not schedule:
        return []
    return schedule.history


def update_schedule_retries(agent_name: str, schedule_id: str, max_retries: int, retry_delay: int = 30) -> bool:
    """Update retry settings for a schedule."""
    cfg = _load_config()

    if agent_name not in cfg.get("agents", {}):
        return False

    for s in cfg["agents"][agent_name].get("schedules", []):
        if s.get("id") == schedule_id:
            s["max_retries"] = max(0, min(max_retries, 3))  # 0-3
            s["retry_delay_seconds"] = max(5, retry_delay)  # Min 5s
            _save_config(cfg)
            return True
    return False


def run_schedule_now(agent_name: str, schedule_id: str, agent_runner=None) -> str:
    """
    Manually trigger a scheduled task immediately.

    Args:
        agent_name: Name of the agent
        schedule_id: ID of the schedule to run
        agent_runner: Callable(agent_name, task) -> str

    Returns:
        The agent's response text, or an error message.
    """
    schedule = get_schedule(agent_name, schedule_id)
    if not schedule:
        return f"Schedule '{schedule_id}' not found for agent '{agent_name}'"

    if not agent_runner:
        return "No agent runner configured"

    start_time = time.monotonic()
    try:
        result = agent_runner(agent_name, schedule.task)
        elapsed = int((time.monotonic() - start_time) * 1000)
        update_schedule_result(
            agent_name, schedule_id, "success",
            elapsed_ms=elapsed, is_catchup=False, is_retry=False,
        )
        return result
    except Exception as e:
        elapsed = int((time.monotonic() - start_time) * 1000)
        update_schedule_result(
            agent_name, schedule_id, f"error: {e}",
            elapsed_ms=elapsed, is_catchup=False, is_retry=False,
        )
        raise
