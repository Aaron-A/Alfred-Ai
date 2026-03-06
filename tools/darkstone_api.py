"""
DarkStone Capital API Tool
Operations tool for monitoring and managing the DarkStone Capital trading app.

Wraps the FastAPI app at localhost:8000 + provides process management.
This is a workspace-local tool for the darkstone-ops agent.
"""

import json
import os
import signal
import subprocess
import time
from datetime import datetime, date
from pathlib import Path
from core.tools import ToolRegistry, ToolParameter
from core.logging import get_logger

logger = get_logger("darkstone_api")

TOOL_META = {
    "version": "1.0.0",
    "author": "Alfred AI",
    "description": "DarkStone Capital trading app operations tool",
    "dependencies": [],
}

BASE_URL = "http://127.0.0.1:8000"
APP_DIR = "/Users/darkstone/Desktop/DarkStoneCapital"
PID_FILE = os.path.join(APP_DIR, ".dashboard.pid")
LOGS_DIR = os.path.join(APP_DIR, "logs")


# ─── HTTP Helpers ────────────────────────────────────────────

def _get(endpoint: str, timeout: int = 15) -> dict | str:
    """GET request to DarkStone Capital API. Returns parsed JSON or error string."""
    import urllib.request
    import urllib.error

    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "Alfred-DarkstoneOps/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return f"Connection error: DarkStone Capital app not reachable at {BASE_URL}. Is it running? ({e.reason})"
    except Exception as e:
        return f"Error fetching {endpoint}: {e}"


def _post(endpoint: str, body: dict = None, timeout: int = 30) -> dict | str:
    """POST request to DarkStone Capital API. Returns parsed JSON or error string."""
    import urllib.request
    import urllib.error

    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(body or {}).encode("utf-8") if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "Alfred-DarkstoneOps/1.0"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if raw.strip():
                return json.loads(raw)
            return {"status": "ok"}
    except urllib.error.URLError as e:
        return f"Connection error: DarkStone Capital app not reachable at {BASE_URL}. ({e.reason})"
    except Exception as e:
        return f"Error posting to {endpoint}: {e}"


# ─── Process Detection Helpers ───────────────────────────────

def _find_pid_on_port(port: int = 8000) -> int | None:
    """Find the PID listening on the given port. Returns PID or None."""
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # May return multiple PIDs; take the first
            return int(result.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def _get_running_pid() -> int | None:
    """
    Get the PID of the running DarkStone Capital app.
    Checks PID file first, falls back to port detection.
    Also fixes stale PID files when the real process is found on the port.
    """
    # Try PID file first
    pid_path = Path(PID_FILE)
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)  # Check if alive
            return pid
        except (ProcessLookupError, ValueError, OSError):
            pass  # PID file is stale

    # Fallback: check what's actually on the port
    port_pid = _find_pid_on_port(8000)
    if port_pid:
        # Update stale PID file so future checks are fast
        try:
            pid_path.write_text(str(port_pid))
            logger.info(f"Updated stale PID file: {port_pid}")
        except OSError:
            pass
    return port_pid


# ─── Action Handlers ─────────────────────────────────────────

def _action_status() -> str:
    """Get agent status (mode, last heartbeat, last action)."""
    result = _get("/api/agent/status")
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2)


def _action_dashboard() -> str:
    """Get full dashboard data: equity, positions, signals, scheduler, P&L."""
    result = _get("/api/data", timeout=20)
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


def _action_positions() -> str:
    """Get current positions with P&L (extracted from dashboard data)."""
    result = _get("/api/data", timeout=20)
    if isinstance(result, str):
        return result
    # Extract position-relevant fields
    positions_data = {
        "equity": result.get("equity"),
        "trades": result.get("trades", []),
        "trade_count": result.get("trade_count", 0),
    }
    return json.dumps(positions_data, indent=2, default=str)


def _action_pulse() -> str:
    """Get current market pulse."""
    result = _get("/api/pulse")
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


def _action_memory() -> str:
    """Get lessons learned from memory.json."""
    result = _get("/api/memory")
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


def _action_principles() -> str:
    """Get confirmed trading rules."""
    result = _get("/api/principles")
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


def _action_thoughts() -> str:
    """Get recent agent decision logs."""
    result = _get("/api/agent/thoughts")
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


def _action_scheduler_state() -> str:
    """Get scheduler state with last runs (extracted from dashboard data)."""
    result = _get("/api/data", timeout=20)
    if isinstance(result, str):
        return result
    scheduler = result.get("scheduler", {})
    return json.dumps(scheduler, indent=2, default=str)


def _action_forecasts() -> str:
    """Get weekly/monthly forecasts."""
    result = _get("/api/forecasts")
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


def _action_trade_limits() -> str:
    """Get per-ticker trade limits."""
    result = _get("/api/trade-limits")
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


def _action_trigger_analysis(params: dict) -> str:
    """Trigger a premarket/midday/aftermarket analysis run."""
    mode = params.get("mode", "midday")
    if mode not in ("premarket", "midday", "aftermarket"):
        return f"Invalid mode '{mode}'. Use: premarket, midday, or aftermarket"

    result = _post(f"/api/scheduler/run/{mode}")
    if isinstance(result, str):
        return result
    return f"Analysis triggered: {mode}\n{json.dumps(result, indent=2, default=str)}"


def _action_trigger_pulse() -> str:
    """Refresh market pulse."""
    result = _post("/api/pulse/run")
    if isinstance(result, str):
        return result
    return f"Pulse refresh triggered.\n{json.dumps(result, indent=2, default=str)}"


def _action_signals_cancel(params: dict) -> str:
    """Cancel a watching signal by ID."""
    signal_id = params.get("signal_id", "")
    if not signal_id:
        return "Error: signal_id is required. Use dashboard action to see active signal IDs."

    result = _post(f"/api/signals/{signal_id}/cancel")
    if isinstance(result, str):
        return result
    return f"Signal {signal_id} cancelled.\n{json.dumps(result, indent=2, default=str)}"


def _action_process_check() -> str:
    """Check if the DarkStone Capital app process is alive."""
    pid = _get_running_pid()
    if pid:
        return f"Process status: ALIVE (PID {pid})"
    else:
        return "Process status: DEAD — app is not running on port 8000."


def _kill_process() -> str | None:
    """Kill the DarkStone Capital app process. Returns error string or None on success."""
    pid = _get_running_pid()
    if not pid:
        return None  # Nothing running

    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        # Force kill if still alive
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            pass
        # Clean up PID file
        try:
            Path(PID_FILE).unlink(missing_ok=True)
        except OSError:
            pass
        return None
    except Exception as e:
        return f"Error killing process (PID {pid}): {e}"


def _start_process() -> str:
    """Start the DarkStone Capital app. Returns status string."""
    run_script = os.path.join(APP_DIR, "run.sh")
    if not os.path.exists(run_script):
        return "Error: run.sh not found in DarkStone Capital directory"

    try:
        proc = subprocess.Popen(
            ["bash", run_script],
            cwd=APP_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent
        )
        time.sleep(3)  # Give it time to start

        # Verify it's alive
        if proc.poll() is None:
            # Check if the HTTP server is responding
            import urllib.request
            try:
                req = urllib.request.Request(f"{BASE_URL}/api/agent/status")
                with urllib.request.urlopen(req, timeout=5):
                    return f"Process started successfully (PID {proc.pid}). App is responding at {BASE_URL}."
            except Exception:
                return f"Process started (PID {proc.pid}) but HTTP server not responding yet. May need a few more seconds."
        else:
            return f"Error: Process exited immediately with code {proc.returncode}"
    except Exception as e:
        return f"Error starting app: {e}"


def _action_process_start() -> str:
    """Start the DarkStone Capital app (only if not already running)."""
    pid = _get_running_pid()
    if pid:
        return f"App is already running (PID {pid}). Use process_restart to restart it."

    return _start_process()


def _action_process_stop() -> str:
    """Stop the DarkStone Capital app."""
    pid = _get_running_pid()
    if not pid:
        return "App is not running."

    err = _kill_process()
    if err:
        return err

    # Verify it's dead
    time.sleep(1)
    still_running = _find_pid_on_port(8000)
    if still_running:
        return f"Warning: Process may still be alive on port 8000 (PID {still_running})."
    return f"App stopped successfully (was PID {pid})."


def _action_process_restart() -> str:
    """Restart the DarkStone Capital app."""
    err = _kill_process()
    if err:
        logger.warning(err)
    return _start_process()


def _action_logs_tail(params: dict) -> str:
    """Read last N lines of dashboard.log."""
    lines = params.get("lines", 50)
    lines = min(max(10, int(lines)), 200)

    log_file = Path(LOGS_DIR) / "dashboard.log"
    if not log_file.exists():
        return "No dashboard.log found"

    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        tail = all_lines[-lines:]
        return f"Last {len(tail)} lines of dashboard.log:\n\n" + "\n".join(tail)
    except Exception as e:
        return f"Error reading logs: {e}"


def _action_logs_errors() -> str:
    """Check for today's error log files."""
    today = date.today().strftime("%Y-%m-%d")
    logs_path = Path(LOGS_DIR)

    if not logs_path.exists():
        return "No logs directory found"

    error_files = list(logs_path.glob(f"{today}_*_error.txt"))
    if not error_files:
        return f"No error logs found for today ({today}). All clear."

    output = [f"Found {len(error_files)} error log(s) for {today}:\n"]
    for ef in error_files:
        try:
            content = ef.read_text(encoding="utf-8", errors="replace")
            name = ef.name
            # Truncate long error files
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            output.append(f"--- {name} ---\n{content}\n")
        except Exception as e:
            output.append(f"--- {ef.name} ---\nError reading: {e}\n")

    return "\n".join(output)


def _action_add_ticker(params: dict) -> str:
    """Add a ticker to the watchlist."""
    ticker = params.get("ticker", "").upper().strip()
    if not ticker:
        return "Error: ticker is required."
    result = _post("/api/actions/add-ticker", {"ticker": ticker})
    if isinstance(result, str):
        return result
    return result.get("result", f"Added {ticker} to watchlist.")


def _action_remove_ticker(params: dict) -> str:
    """Remove a ticker from the watchlist."""
    ticker = params.get("ticker", "").upper().strip()
    if not ticker:
        return "Error: ticker is required."
    result = _post("/api/actions/remove-ticker", {"ticker": ticker})
    if isinstance(result, str):
        return result
    return result.get("result", f"Removed {ticker} from watchlist.")


def _action_set_watch_only(params: dict) -> str:
    """Set a ticker to watch-only mode."""
    ticker = params.get("ticker", "").upper().strip()
    if not ticker:
        return "Error: ticker is required."
    result = _post("/api/actions/set-watch-only", {"ticker": ticker})
    if isinstance(result, str):
        return result
    return result.get("result", f"Set {ticker} to watch-only.")


def _action_tighten_focus() -> str:
    """Cancel medium-conviction signals to tighten focus."""
    result = _post("/api/actions/tighten-focus")
    if isinstance(result, str):
        return result
    return result.get("result", "Tightened focus — cancelled medium-conviction signals.")


def _action_think_now() -> str:
    """Force an immediate agent heartbeat."""
    result = _post("/api/actions/think-now")
    if isinstance(result, str):
        return result
    return result.get("result", "Forced agent heartbeat.")


def _action_research() -> str:
    """Trigger a research scan."""
    result = _post("/api/actions/research", timeout=60)
    if isinstance(result, str):
        return result
    return result.get("result", "Research scan triggered.")


# ─── Main Tool Function ─────────────────────────────────────

# Action dispatch table
_ACTIONS = {
    "status": lambda p: _action_status(),
    "dashboard": lambda p: _action_dashboard(),
    "positions": lambda p: _action_positions(),
    "pulse": lambda p: _action_pulse(),
    "memory": lambda p: _action_memory(),
    "principles": lambda p: _action_principles(),
    "thoughts": lambda p: _action_thoughts(),
    "scheduler_state": lambda p: _action_scheduler_state(),
    "forecasts": lambda p: _action_forecasts(),
    "trade_limits": lambda p: _action_trade_limits(),
    "trigger_analysis": lambda p: _action_trigger_analysis(p),
    "trigger_pulse": lambda p: _action_trigger_pulse(),
    "signals_cancel": lambda p: _action_signals_cancel(p),
    "process_check": lambda p: _action_process_check(),
    "process_start": lambda p: _action_process_start(),
    "process_stop": lambda p: _action_process_stop(),
    "process_restart": lambda p: _action_process_restart(),
    "logs_tail": lambda p: _action_logs_tail(p),
    "logs_errors": lambda p: _action_logs_errors(),
    "add_ticker": lambda p: _action_add_ticker(p),
    "remove_ticker": lambda p: _action_remove_ticker(p),
    "set_watch_only": lambda p: _action_set_watch_only(p),
    "tighten_focus": lambda p: _action_tighten_focus(),
    "think_now": lambda p: _action_think_now(),
    "research": lambda p: _action_research(),
}


def darkstone_api(action: str, params: str = None) -> str:
    """
    Interact with the DarkStone Capital trading application.

    Routes to the appropriate action handler based on the action parameter.
    """
    action = action.strip().lower()

    if action not in _ACTIONS:
        available = ", ".join(sorted(_ACTIONS.keys()))
        return f"Unknown action '{action}'. Available actions: {available}"

    # Parse params if provided
    parsed_params = {}
    if params:
        try:
            parsed_params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            # Try treating it as a simple key=value
            parsed_params = {"value": params}

    try:
        result = _ACTIONS[action](parsed_params)
        # Truncate very large results
        if len(result) > 15000:
            result = result[:15000] + "\n... (truncated, output too large)"
        return result
    except Exception as e:
        logger.error(f"darkstone_api({action}) error: {e}")
        return f"Error executing {action}: {e}"


# ─── Tool Registration ──────────────────────────────────────

def register(registry: ToolRegistry):
    """Register the DarkStone Capital API tool."""
    registry.register_function(
        name="darkstone_api",
        description=(
            "Interact with the DarkStone Capital trading application at localhost:8000. "
            "Actions: status (scheduler status), dashboard (full state with equity/positions/P&L), "
            "positions (open positions), pulse (market bias), memory (lessons learned), "
            "principles (trading rules), thoughts (agent decisions), scheduler_state (last runs), "
            "forecasts (weekly/monthly outlook), trade_limits (per-ticker limits), "
            "trigger_analysis (run premarket/midday/aftermarket), trigger_pulse (refresh pulse), "
            "signals_cancel (cancel a signal), process_check (is app alive?), "
            "process_start (start the app), process_stop (stop the app), "
            "process_restart (restart the app), logs_tail (recent log lines), "
            "logs_errors (today's error files), "
            "add_ticker (add ticker to watchlist), remove_ticker (remove from watchlist), "
            "set_watch_only (set ticker watch-only), tighten_focus (cancel medium-conviction signals), "
            "think_now (force agent heartbeat), research (trigger research scan). "
            "Use the 'action' parameter to specify what to do."
        ),
        fn=darkstone_api,
        parameters=[
            ToolParameter("action", "string",
                "Action to perform: status, dashboard, positions, pulse, memory, "
                "principles, thoughts, scheduler_state, forecasts, trade_limits, "
                "trigger_analysis, trigger_pulse, signals_cancel, process_check, "
                "process_start, process_stop, process_restart, "
                "logs_tail, logs_errors, add_ticker, remove_ticker, "
                "set_watch_only, tighten_focus, think_now, research"),
            ToolParameter("params", "string",
                "JSON string of parameters for actions that need them. "
                "E.g. trigger_analysis: '{\"mode\": \"midday\"}', "
                "signals_cancel: '{\"signal_id\": \"abc123\"}', "
                "add_ticker: '{\"ticker\": \"AAPL\"}', "
                "logs_tail: '{\"lines\": 100}'",
                required=False),
        ],
        category="trading",
        source="workspace",
        file_path=__file__,
    )
