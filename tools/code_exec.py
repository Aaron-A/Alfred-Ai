"""
Code Execution Tool
Execute Python, bash, or Node.js code and return output.

Agents can write and run code on the fly — data processing, calculations,
file manipulation, API calls, or any task that benefits from actual code.
"""

import os
import re
import uuid
import subprocess
import shutil
import time
from pathlib import Path
from core.tools import ToolRegistry, ToolParameter
from core.config import config
from core.logging import get_logger

logger = get_logger("code_exec")

TOOL_META = {
    "version": "0.1.0",
    "author": "Alfred AI",
    "description": "Execute Python, bash, or Node.js code with output capture and timeout protection",
    "dependencies": [],
}

MAX_OUTPUT_CHARS = 10000
MAX_CODE_CHARS = 50000

_LANGUAGE_CONFIG = {
    "python": {
        "executable": None,  # Resolved at runtime
        "extension": ".py",
        "args": [],
    },
    "bash": {
        "executable": "/bin/bash",
        "extension": ".sh",
        "args": [],
    },
    "node": {
        "executable": None,  # Resolved at runtime
        "extension": ".js",
        "args": [],
    },
}

# Dangerous patterns blocked in Python code (best-effort safety net)
_BLOCKED_PYTHON_PATTERNS = [
    (r"os\.fork\s*\(", "os.fork() is not allowed"),
    (r"while\s+True.*os\.(system|popen|exec)", "Infinite loop with shell exec is not allowed"),
    (r"shutil\.rmtree\s*\(\s*['\"/]", "shutil.rmtree on absolute paths is not allowed"),
    (r"rm\s+-rf\s+/", "rm -rf / is not allowed"),
    (r"socket\..*\.bind\s*\(", "Binding network sockets is not allowed"),
    (r"socketserver\.", "Socket servers are not allowed"),
    (r"http\.server", "HTTP servers are not allowed"),
    (r"ctypes\.", "ctypes is not allowed"),
    (r":\s*\(\s*\)\s*\{", "Fork bombs are not allowed"),
]


def _get_python_executable() -> str:
    """Get the project's venv Python interpreter."""
    venv_python = config.PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python3"


def _get_temp_dir() -> Path:
    """Get temp directory for code execution, scoped to workspace."""
    ws = os.environ.get("ALFRED_WORKSPACE", "")
    if ws:
        temp_dir = Path(ws) / ".code_exec"
    else:
        temp_dir = config.PROJECT_ROOT / "data" / ".code_exec"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _scan_python_code(code: str) -> str | None:
    """Scan Python code for dangerous patterns. Returns violation message or None."""
    for pattern, message in _BLOCKED_PYTHON_PATTERNS:
        if re.search(pattern, code):
            return message
    return None


def _truncate_output(text: str) -> str:
    """Truncate output to max chars."""
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(text)} chars total)"
    return text


def code_exec(code: str, language: str = "python", timeout: int = 30) -> str:
    """Execute code and return stdout + stderr + exit code."""
    language = language.lower().strip()
    if language not in _LANGUAGE_CONFIG:
        return f"Error: Unsupported language '{language}'. Use: python, bash, node"

    if not code.strip():
        return "Error: Empty code."

    if len(code) > MAX_CODE_CHARS:
        return f"Error: Code too long ({len(code)} chars, max {MAX_CODE_CHARS})."

    timeout = min(max(5, timeout), 120)

    # Security scan for Python
    if language == "python":
        violation = _scan_python_code(code)
        if violation:
            return f"Blocked: {violation}"

    # Resolve executable
    lang_cfg = _LANGUAGE_CONFIG[language]
    if language == "python":
        executable = _get_python_executable()
    elif language == "node":
        executable = shutil.which("node")
        if not executable:
            return "Error: Node.js not found on this system."
    else:
        executable = lang_cfg["executable"]

    # Write temp file
    temp_dir = _get_temp_dir()
    temp_file = temp_dir / f"exec_{uuid.uuid4().hex[:8]}{lang_cfg['extension']}"

    # Working directory
    ws = os.environ.get("ALFRED_WORKSPACE", "")
    cwd = ws if ws else str(config.PROJECT_ROOT)

    try:
        temp_file.write_text(code, encoding="utf-8")
        cmd = [executable] + lang_cfg["args"] + [str(temp_file)]

        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env={
                **dict(os.environ),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        elapsed = time.time() - start

        # Format output
        lines = [f"Language: {language} | Exit: {result.returncode} | {elapsed:.1f}s"]

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if stdout:
            lines.append(f"\n--- stdout ---\n{_truncate_output(stdout)}")

        if stderr:
            lines.append(f"\n--- stderr ---\n{_truncate_output(stderr)}")

        if not stdout and not stderr:
            lines.append("(no output)")

        return "\n".join(lines)

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            pass


# ─── Tool Registration ──────────────────────────────────────────

def register(registry: ToolRegistry):
    """Register code execution tool."""
    registry.register_function(
        name="code_exec",
        description=(
            "Execute code (Python, bash, or Node.js) and return the output. "
            "Python runs in Alfred's virtual environment with access to all installed packages. "
            "Use this for data processing, calculations, file manipulation, API calls, "
            "or any task that benefits from running actual code. "
            "Working directory is the agent's workspace. "
            "Returns stdout, stderr, and exit code. Output truncated at 10KB."
        ),
        fn=code_exec,
        parameters=[
            ToolParameter("code", "string", "The code to execute"),
            ToolParameter("language", "string",
                "Programming language: python (default), bash, node",
                required=False, enum=["python", "bash", "node"]),
            ToolParameter("timeout", "integer",
                "Execution timeout in seconds (default 30, max 120)",
                required=False),
        ],
        category="code",
        source="shared",
        file_path=__file__,
    )
