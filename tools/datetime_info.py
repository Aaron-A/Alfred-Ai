"""
DateTime Tool
Get current date/time, convert between timezones, and calculate date differences.

LLMs often don't know the current date/time — this tool fills that gap.
Uses only stdlib (datetime, zoneinfo). No external dependencies.
"""

from datetime import datetime, timezone as _tz, timedelta
from core.tools import ToolRegistry, ToolParameter
from core.config import config
from core.logging import get_logger

logger = get_logger("datetime_info")


def register(registry: ToolRegistry):
    """Register datetime tools."""
    registry.register_function(
        name="datetime_info",
        description=(
            "Get the current date and time, or convert between timezones. "
            "Returns date, time, day of week, unix timestamp, and more. "
            "Use this whenever you need to know what time it is or work with dates."
        ),
        fn=datetime_info,
        parameters=[
            ToolParameter(
                "timezone", "string",
                "Timezone name (e.g. 'US/Eastern', 'Europe/London', 'Asia/Tokyo'). Default: Alfred's configured timezone",
                required=False,
            ),
            ToolParameter(
                "format", "string",
                "Output format: 'full' (default), 'date', 'time', 'iso', 'unix'",
                required=False,
            ),
        ],
        category="utility",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="date_diff",
        description=(
            "Calculate the difference between two dates. "
            "Returns days, weeks, and a human-readable duration. "
            "Dates should be in YYYY-MM-DD format."
        ),
        fn=date_diff,
        parameters=[
            ToolParameter("date1", "string", "First date in YYYY-MM-DD format"),
            ToolParameter("date2", "string", "Second date in YYYY-MM-DD format"),
        ],
        category="utility",
        source="shared",
        file_path=__file__,
    )


def datetime_info(timezone: str = None, format: str = "full") -> str:
    """Get current date/time info."""
    if timezone is None:
        timezone = config.TIMEZONE
    try:
        tz = _get_timezone(timezone)
    except Exception as e:
        return f"Error: {e}"

    now = datetime.now(tz)

    if format == "iso":
        return now.isoformat()
    elif format == "unix":
        return str(int(now.timestamp()))
    elif format == "date":
        return now.strftime("%Y-%m-%d")
    elif format == "time":
        return now.strftime("%H:%M:%S %Z")

    # Full format
    utc_now = datetime.now(_tz.utc)
    lines = [
        f"Current time ({timezone}):",
        f"  Date:      {now.strftime('%Y-%m-%d')}",
        f"  Time:      {now.strftime('%H:%M:%S %Z')}",
        f"  Day:       {now.strftime('%A')}",
        f"  Week:      {now.strftime('%W')} of {now.year}",
        f"  ISO:       {now.isoformat()}",
        f"  Unix:      {int(now.timestamp())}",
    ]

    # Add UTC if not already UTC
    if timezone.upper() != "UTC":
        lines.append(f"  UTC:       {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)


def date_diff(date1: str, date2: str) -> str:
    """Calculate difference between two dates."""
    try:
        d1 = datetime.strptime(date1.strip(), "%Y-%m-%d")
    except ValueError:
        return f"Error: Can't parse '{date1}' — use YYYY-MM-DD format."

    try:
        d2 = datetime.strptime(date2.strip(), "%Y-%m-%d")
    except ValueError:
        return f"Error: Can't parse '{date2}' — use YYYY-MM-DD format."

    diff = d2 - d1
    days = diff.days
    abs_days = abs(days)

    # Human-readable
    years = abs_days // 365
    remaining = abs_days % 365
    months = remaining // 30
    remaining_days = remaining % 30
    weeks = abs_days // 7

    parts = []
    if years:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    if remaining_days:
        parts.append(f"{remaining_days} day{'s' if remaining_days != 1 else ''}")

    direction = "later" if days > 0 else "earlier" if days < 0 else "same day"
    human = ", ".join(parts) if parts else "same day"

    lines = [
        f"From {date1} to {date2}:",
        f"  Days:     {abs_days} ({direction})",
        f"  Weeks:    {weeks}",
        f"  Duration: {human}",
    ]
    return "\n".join(lines)


def _get_timezone(name: str):
    """Get a timezone object by name. Tries zoneinfo first, falls back to UTC offset."""
    if not name or name.upper() == "UTC":
        return _tz.utc

    # Try zoneinfo (Python 3.9+)
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except ImportError:
        pass
    except KeyError:
        pass

    # Common abbreviation fallbacks
    _offsets = {
        "EST": -5, "EDT": -4,
        "CST": -6, "CDT": -5,
        "MST": -7, "MDT": -6,
        "PST": -8, "PDT": -7,
        "GMT": 0, "BST": 1,
        "CET": 1, "CEST": 2,
        "JST": 9, "KST": 9,
        "IST": 5.5, "AEST": 10, "AEDT": 11,
    }

    upper = name.upper()
    if upper in _offsets:
        hours = _offsets[upper]
        return _tz(timedelta(hours=hours))

    raise ValueError(
        f"Unknown timezone '{name}'. Use IANA names like 'US/Eastern', "
        f"'Europe/London', 'Asia/Tokyo', or abbreviations like 'EST', 'PST'."
    )
