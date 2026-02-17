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
from dataclasses import dataclass, field, asdict

from .config import _load_config, _save_config


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

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

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
        dt = datetime.now()

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
        time_str = f"{int(hour):02d}:{int(minute):02d}"
    elif minute.startswith("*/"):
        time_str = f"every {minute[2:]} min"
    elif hour.startswith("*/"):
        time_str = f"every {hour[2:]} hr"

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


def update_schedule_result(agent_name: str, schedule_id: str, result: str):
    """Update the last_run and last_result for a schedule."""
    cfg = _load_config()

    if agent_name not in cfg.get("agents", {}):
        return

    for s in cfg["agents"][agent_name].get("schedules", []):
        if s.get("id") == schedule_id:
            s["last_run"] = datetime.now().isoformat()
            s["last_result"] = result[:200]  # Truncate
            _save_config(cfg)
            return


# ─── Scheduler Loop ─────────────────────────────────────────────

class Scheduler:
    """
    Background scheduler that checks for due tasks every minute.

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

    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _loop(self):
        """Main scheduler loop — checks every 30 seconds."""
        while self._running:
            now = datetime.now()

            # Only check once per minute
            current_minute = now.hour * 60 + now.minute
            if current_minute != self._last_check_minute:
                self._last_check_minute = current_minute
                self._check_schedules(now)

            # Sleep 30 seconds between checks
            time.sleep(30)

    def _check_schedules(self, now: datetime):
        """Check all agent schedules and run any that are due."""
        cfg = _load_config()

        for agent_name, agent_cfg in cfg.get("agents", {}).items():
            # Skip paused agents
            if agent_cfg.get("status") == "paused":
                continue

            for schedule_data in agent_cfg.get("schedules", []):
                schedule = Schedule.from_dict(schedule_data)

                if not schedule.enabled:
                    continue

                if cron_matches(schedule.cron, now):
                    # Check if we already ran this minute
                    if schedule.last_run:
                        try:
                            last = datetime.fromisoformat(schedule.last_run)
                            if (last.hour == now.hour
                                    and last.minute == now.minute
                                    and last.date() == now.date()):
                                continue  # Already ran this minute
                        except (ValueError, TypeError):
                            pass

                    self._execute_schedule(agent_name, schedule)

    def _execute_schedule(self, agent_name: str, schedule: Schedule):
        """Execute a scheduled task."""
        print(f"  [scheduler] Running '{schedule.id}' for {agent_name}: {schedule.task[:60]}")

        if self._runner:
            try:
                result = self._runner(agent_name, schedule.task)
                update_schedule_result(agent_name, schedule.id, "success")
                print(f"  [scheduler] '{schedule.id}' completed")
            except Exception as e:
                update_schedule_result(agent_name, schedule.id, f"error: {e}")
                print(f"  [scheduler] '{schedule.id}' failed: {e}")
        else:
            update_schedule_result(agent_name, schedule.id, "no runner configured")
            print(f"  [scheduler] No agent runner configured, skipping execution")


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
