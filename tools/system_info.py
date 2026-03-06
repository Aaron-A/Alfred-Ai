"""
System Info Tool
Reports CPU, memory, disk, uptime, and network stats for the host machine.
"""

import os
import platform
import subprocess
import time
from datetime import timedelta
from core.tools import ToolRegistry, ToolParameter
from core.logging import get_logger

logger = get_logger("system_info")

TOOL_META = {
    "version": "1.0.0",
    "author": "Alfred AI",
    "description": "Host machine system status tool",
    "dependencies": [],
}


def _run(cmd: list[str], timeout: int = 5) -> str:
    """Run a command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception:
        return ""


def system_status() -> str:
    """Get comprehensive system status: CPU, RAM, disk, uptime, load."""
    lines = []

    # ── Host Info ──
    uname = platform.uname()
    lines.append(f"Host: {uname.node}")
    lines.append(f"OS: {uname.system} {uname.release} ({uname.machine})")

    # ── Uptime ──
    uptime_raw = _run(["uptime"])
    if uptime_raw:
        lines.append(f"Uptime: {uptime_raw.strip()}")

    # ── CPU ──
    # macOS: use sysctl for CPU info
    cpu_brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    cpu_cores = _run(["sysctl", "-n", "hw.ncpu"])
    if cpu_brand:
        lines.append(f"CPU: {cpu_brand} ({cpu_cores} cores)")

    # Load averages
    load = os.getloadavg()
    lines.append(f"Load: {load[0]:.2f} (1m) / {load[1]:.2f} (5m) / {load[2]:.2f} (15m)")

    # CPU usage via top (macOS)
    top_out = _run(["top", "-l", "1", "-n", "0", "-s", "0"])
    if top_out:
        for line in top_out.splitlines():
            if "CPU usage" in line:
                lines.append(f"CPU: {line.strip()}")
                break

    # ── Memory ──
    if platform.system() == "Darwin":
        # macOS: parse vm_stat
        vm = _run(["vm_stat"])
        page_size = 16384  # macOS default
        try:
            ps_out = _run(["sysctl", "-n", "hw.pagesize"])
            if ps_out:
                page_size = int(ps_out)
        except Exception:
            pass

        total_mem = _run(["sysctl", "-n", "hw.memsize"])
        if total_mem:
            total_gb = int(total_mem) / (1024 ** 3)
        else:
            total_gb = 0

        if vm:
            stats = {}
            for line in vm.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    val = val.strip().rstrip(".")
                    try:
                        stats[key.strip()] = int(val)
                    except ValueError:
                        pass

            free_pages = stats.get("Pages free", 0)
            active_pages = stats.get("Pages active", 0)
            inactive_pages = stats.get("Pages inactive", 0)
            wired_pages = stats.get("Pages wired down", 0)
            compressed = stats.get("Pages occupied by compressor", 0)

            used_gb = (active_pages + wired_pages + compressed) * page_size / (1024 ** 3)
            free_gb = total_gb - used_gb if total_gb else 0
            pct = (used_gb / total_gb * 100) if total_gb else 0

            lines.append(f"RAM: {used_gb:.1f}G used / {total_gb:.0f}G total ({pct:.0f}%)")
    else:
        # Linux: /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            total = available = 0
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / (1024 * 1024)
            used = total - available
            pct = (used / total * 100) if total else 0
            lines.append(f"RAM: {used:.1f}G used / {total:.0f}G total ({pct:.0f}%)")
        except Exception:
            pass

    # ── Disk ──
    df_out = _run(["df", "-h", "/"])
    if df_out:
        # Parse the second line of df output
        df_lines = df_out.strip().splitlines()
        if len(df_lines) >= 2:
            parts = df_lines[1].split()
            if len(parts) >= 5:
                lines.append(f"Disk (/): {parts[2]} used / {parts[1]} total ({parts[4]} full)")

    # ── Key Processes ──
    # Alfred: check port 7700 (its API server) — more reliable than pgrep from within
    alfred_pid = _run(["lsof", "-i", ":7700", "-t", "-sTCP:LISTEN"])
    if alfred_pid:
        lines.append(f"Alfred: Running (PID {alfred_pid.splitlines()[0]}, port 7700)")
    else:
        # Fallback: check process list
        alfred_pid = _run(["pgrep", "-f", "alfred-ai/__main__.py"])
        if alfred_pid:
            lines.append(f"Alfred: Running (PID {alfred_pid.splitlines()[0]})")
        else:
            lines.append("Alfred: Not detected")

    # DarkStone Capital: check port 8000
    ds_pid = _run(["lsof", "-i", ":8000", "-t", "-sTCP:LISTEN"])
    if ds_pid:
        lines.append(f"DarkStone Capital: Running (PID {ds_pid.splitlines()[0]}, port 8000)")
    else:
        lines.append("DarkStone Capital: Not running")

    return "\n".join(lines)


# ─── Tool Registration ──────────────────────────────────────

def register(registry: ToolRegistry):
    """Register the system info tool."""
    registry.register_function(
        name="system_status",
        description=(
            "Get system status of the host machine Alfred is running on. "
            "Returns CPU info and usage, RAM usage, disk usage, uptime, load averages, "
            "and status of key processes (Alfred, DarkStone Capital app)."
        ),
        fn=system_status,
        parameters=[],
        category="system",
        source="shared",
        file_path=__file__,
    )
