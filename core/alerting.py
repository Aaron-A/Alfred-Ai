"""
Alfred AI — Alert System
Discord webhook-based alerting for errors, cost thresholds, and bot crashes.

Alert rules are defined in alfred.json:
{
    "alerts": {
        "discord_webhook": "https://discord.com/api/webhooks/...",
        "rules": [
            {"type": "error_rate", "threshold": 3, "window_minutes": 60},
            {"type": "daily_cost", "threshold": 10.0},
            {"type": "bot_crash", "enabled": true}
        ],
        "cooldown_minutes": 60
    }
}
"""

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .config import _load_config, config
from .logging import get_logger

logger = get_logger("alerting")

# Track when each alert type was last fired (cooldown)
_last_fired: dict[str, datetime] = {}


def _get_alert_config() -> dict:
    """Load alert configuration from alfred.json."""
    cfg = _load_config()
    return cfg.get("alerts", {})


def _should_fire(alert_type: str, cooldown_minutes: int = 60) -> bool:
    """Check if enough time has passed since this alert type last fired."""
    last = _last_fired.get(alert_type)
    if last is None:
        return True
    elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 60
    return elapsed >= cooldown_minutes


def _send_discord_alert(webhook_url: str, title: str, description: str, color: int = 0xFF4444):
    """Send an alert to a Discord webhook."""
    if not webhook_url:
        return

    embed = {
        "title": f"🚨 {title}",
        "description": description,
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "Alfred AI Alerting"},
    }

    payload = json.dumps({"embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Alfred-AI/1.0"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
        logger.info(f"Alert sent: {title}")
    except Exception as e:
        logger.warning(f"Failed to send alert: {e}")


def check_error_alert(agent: str, error: str):
    """
    Check if error rate exceeds threshold and fire alert if needed.

    Called after each metrics.record_error(). Counts recent errors
    and alerts if threshold is exceeded.
    """
    alert_cfg = _get_alert_config()
    if not alert_cfg:
        return

    webhook = alert_cfg.get("discord_webhook", "")
    if not webhook:
        return

    cooldown = alert_cfg.get("cooldown_minutes", 60)
    if not _should_fire("error_rate", cooldown):
        return

    rules = alert_cfg.get("rules", [])
    for rule in rules:
        if rule.get("type") != "error_rate":
            continue

        threshold = rule.get("threshold", 3)
        window = rule.get("window_minutes", 60)

        # Count recent errors from metrics DB
        try:
            import sqlite3
            db_path = config.DATA_DIR / "metrics.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                """SELECT COUNT(*) FROM events
                   WHERE is_error = 1 AND agent = ?
                   AND timestamp >= datetime('now', ?)""",
                (agent, f"-{window} minutes"),
            )
            count = cursor.fetchone()[0]
            conn.close()

            if count >= threshold:
                _last_fired["error_rate"] = datetime.now(timezone.utc)
                _send_discord_alert(
                    webhook,
                    f"Error Rate Alert — {agent}",
                    f"**{count} errors** in the last {window} minutes (threshold: {threshold})\n\n"
                    f"Latest: `{error[:200]}`",
                )
        except Exception as e:
            logger.debug(f"Error alert check failed: {e}")
        break


def check_cost_alert():
    """
    Check if daily cost exceeds threshold and fire alert.

    Called periodically (e.g., after each agent run).
    """
    alert_cfg = _get_alert_config()
    if not alert_cfg:
        return

    webhook = alert_cfg.get("discord_webhook", "")
    if not webhook:
        return

    cooldown = alert_cfg.get("cooldown_minutes", 60)
    if not _should_fire("daily_cost", cooldown):
        return

    rules = alert_cfg.get("rules", [])
    for rule in rules:
        if rule.get("type") != "daily_cost":
            continue

        threshold = rule.get("threshold", 10.0)

        try:
            import sqlite3
            db_path = config.DATA_DIR / "metrics.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                """SELECT COALESCE(SUM(estimated_cost), 0) FROM events
                   WHERE timestamp >= datetime('now', '-1 day')""",
            )
            total_cost = cursor.fetchone()[0]
            conn.close()

            if total_cost >= threshold:
                _last_fired["daily_cost"] = datetime.now(timezone.utc)
                _send_discord_alert(
                    webhook,
                    "Daily Cost Alert",
                    f"**${total_cost:.2f}** spent in the last 24 hours (threshold: ${threshold:.2f})",
                    color=0xFFA500,  # Orange
                )
        except Exception as e:
            logger.debug(f"Cost alert check failed: {e}")
        break


def send_bot_crash_alert(agent: str, error: str):
    """Send an immediate alert when a bot/agent crashes."""
    alert_cfg = _get_alert_config()
    if not alert_cfg:
        return

    webhook = alert_cfg.get("discord_webhook", "")
    if not webhook:
        return

    # Bot crash alerts always fire (no cooldown check — these are critical)
    rules = alert_cfg.get("rules", [])
    for rule in rules:
        if rule.get("type") == "bot_crash" and rule.get("enabled", True):
            _send_discord_alert(
                webhook,
                f"Bot Crash — {agent}",
                f"The agent/bot crashed with:\n```\n{error[:500]}\n```",
                color=0xFF0000,  # Red
            )
            break
