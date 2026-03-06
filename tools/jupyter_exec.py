"""
Persistent Code Execution Tool (Jupyter Kernel)
Execute Python code in a persistent kernel — variables, imports, and state
survive across calls. Like having an interactive Python session.

Unlike code_exec which starts fresh every time, jupyter_exec maintains
a running IPython kernel so agents can:
- Import a library once and use it in later calls
- Build up data structures across multiple steps
- Iterate on analysis without re-running everything
"""

import os
import time
import atexit
import queue
from pathlib import Path
from core.tools import ToolRegistry, ToolParameter
from core.config import config
from core.logging import get_logger

logger = get_logger("jupyter_exec")

TOOL_META = {
    "version": "0.1.0",
    "author": "Alfred AI",
    "description": "Persistent Python execution via Jupyter kernel — state survives across calls",
    "dependencies": ["ipykernel", "jupyter_client"],
}

MAX_OUTPUT_CHARS = 15000


# ─── Kernel Manager ──────────────────────────────────────────

class _KernelSession:
    """Manages a persistent Jupyter kernel."""

    def __init__(self):
        from jupyter_client import KernelManager

        self._km = KernelManager(kernel_name="python3")
        self._km.start_kernel(
            extra_arguments=["--no-stderr"],
            env={
                **dict(os.environ),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        self._kc = self._km.client()
        self._kc.start_channels()

        # Wait for kernel to be ready
        try:
            self._kc.wait_for_ready(timeout=15)
        except RuntimeError:
            self.close()
            raise RuntimeError("Jupyter kernel failed to start")

        # Set working directory
        ws = os.environ.get("ALFRED_WORKSPACE", "")
        cwd = ws if ws else str(config.PROJECT_ROOT)
        self._kc.execute(f"import os; os.chdir({cwd!r})", silent=True)

        logger.info("Jupyter kernel started")

    @property
    def client(self):
        return self._kc

    def is_alive(self) -> bool:
        return self._km.is_alive()

    def restart(self):
        """Restart the kernel (clears all state)."""
        self._km.restart_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=15)
        logger.info("Jupyter kernel restarted")

    def close(self):
        try:
            self._kc.stop_channels()
            self._km.shutdown_kernel(now=True)
        except Exception:
            pass
        logger.info("Jupyter kernel closed")


_kernel: _KernelSession | None = None


def _get_kernel() -> _KernelSession:
    """Get or create the persistent kernel."""
    global _kernel
    if _kernel is None or not _kernel.is_alive():
        if _kernel is not None:
            try:
                _kernel.close()
            except Exception:
                pass
        _kernel = _KernelSession()
    return _kernel


def _close_kernel():
    """Cleanup on exit."""
    global _kernel
    if _kernel:
        _kernel.close()
        _kernel = None


atexit.register(_close_kernel)


# ─── Execution ───────────────────────────────────────────────

def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(text)} chars total)"
    return text


def jupyter_exec(code: str, timeout: int = 60) -> str:
    """Execute Python code in the persistent Jupyter kernel."""
    if not code.strip():
        return "Error: Empty code."

    if len(code) > 50000:
        return f"Error: Code too long ({len(code)} chars, max 50000)."

    timeout = min(max(5, timeout), 300)

    try:
        kernel = _get_kernel()
    except RuntimeError as e:
        return f"Error starting kernel: {e}"

    kc = kernel.client

    # Execute
    msg_id = kc.execute(code)

    # Collect output
    outputs = []
    errors = []
    result_data = None

    deadline = time.time() + timeout

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return f"Error: Execution timed out after {timeout}s"

        try:
            msg = kc.get_iopub_msg(timeout=min(remaining, 5))
        except queue.Empty:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["msg_type"]
        content = msg["content"]

        if msg_type == "stream":
            outputs.append(content.get("text", ""))

        elif msg_type == "execute_result":
            data = content.get("data", {})
            result_data = data.get("text/plain", "")

        elif msg_type == "display_data":
            data = content.get("data", {})
            if "text/plain" in data:
                outputs.append(data["text/plain"])

        elif msg_type == "error":
            ename = content.get("ename", "Error")
            evalue = content.get("evalue", "")
            traceback_lines = content.get("traceback", [])
            # Clean ANSI escape codes from traceback
            import re
            clean_tb = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in traceback_lines]
            errors.append(f"{ename}: {evalue}\n" + "\n".join(clean_tb[-5:]))

        elif msg_type == "status":
            if content.get("execution_state") == "idle":
                break

    # Format output
    parts = []

    stdout = "".join(outputs).strip()
    if stdout:
        parts.append(stdout)

    if result_data:
        if parts:
            parts.append(f">>> {result_data}")
        else:
            parts.append(result_data)

    if errors:
        parts.append("\n".join(errors))

    if not parts:
        parts.append("(no output)")

    return _truncate("\n".join(parts))


def jupyter_reset() -> str:
    """Restart the Jupyter kernel — clears all variables and imports."""
    global _kernel
    if _kernel and _kernel.is_alive():
        _kernel.restart()
        return "Kernel restarted. All variables and imports cleared."
    else:
        _kernel = None
        _get_kernel()
        return "New kernel started."


def jupyter_vars() -> str:
    """List all user-defined variables in the current kernel session."""
    return jupyter_exec("""
import json
_vars = {}
for _name in sorted(dir()):
    if _name.startswith('_') or _name in ('In', 'Out', 'get_ipython', 'exit', 'quit', 'json'):
        continue
    _obj = eval(_name)
    _type = type(_obj).__name__
    try:
        _repr = repr(_obj)[:100]
    except:
        _repr = '...'
    _vars[_name] = f"{_type}: {_repr}"
if _vars:
    print("\\n".join(f"  {k} = {v}" for k, v in _vars.items()))
else:
    print("(no user variables)")
del _vars, _name, _obj, _type, _repr
""")


# ─── Tool Registration ──────────────────────────────────────────

def register(registry: ToolRegistry):
    """Register persistent Jupyter execution tools."""

    registry.register_function(
        name="jupyter_exec",
        description=(
            "Execute Python code in a PERSISTENT Jupyter kernel. Unlike code_exec, "
            "variables, imports, and state survive between calls. Use this when you "
            "need to build up analysis step by step, import libraries once and reuse "
            "them, or maintain state across multiple operations. "
            "Working directory is the agent's workspace. Max timeout 300s."
        ),
        fn=jupyter_exec,
        parameters=[
            ToolParameter("code", "string", "Python code to execute"),
            ToolParameter("timeout", "integer",
                "Execution timeout in seconds (default 60, max 300)",
                required=False),
        ],
        category="code",
        source="shared",
        file_path=__file__,
        dependencies=["ipykernel", "jupyter_client"],
    )

    registry.register_function(
        name="jupyter_reset",
        description=(
            "Restart the Jupyter kernel — clears all variables, imports, and state. "
            "Use when you want a fresh Python session."
        ),
        fn=jupyter_reset,
        parameters=[],
        category="code",
        source="shared",
        file_path=__file__,
        dependencies=["ipykernel", "jupyter_client"],
    )

    registry.register_function(
        name="jupyter_vars",
        description=(
            "List all user-defined variables in the current Jupyter session. "
            "Shows variable names, types, and values."
        ),
        fn=jupyter_vars,
        parameters=[],
        category="code",
        source="shared",
        file_path=__file__,
        dependencies=["ipykernel", "jupyter_client"],
    )
