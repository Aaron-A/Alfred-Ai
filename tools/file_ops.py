"""
File Operations Tool
Read and write files within the agent's workspace directory.

Safety: paths are sandboxed to the agent's workspace — no traversal outside it.
Agents can use this to persist notes, data, configs, or intermediate results.
"""

import os
from pathlib import Path
from core.tools import ToolRegistry, ToolParameter
from core.logging import get_logger

logger = get_logger("file_ops")


def register(registry: ToolRegistry):
    """Register file read/write tools."""
    registry.register_function(
        name="file_read",
        description=(
            "Read a file from your workspace directory. "
            "Path is relative to your workspace root (e.g. 'notes.md', 'data/prices.json'). "
            "Cannot read files outside your workspace. "
            "Returns the file content as text (max 20000 chars)."
        ),
        fn=file_read,
        parameters=[
            ToolParameter("path", "string", "File path relative to workspace (e.g. 'notes.md')"),
            ToolParameter(
                "max_chars", "integer",
                "Maximum characters to return (default 20000)",
                required=False,
            ),
        ],
        category="filesystem",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="file_write",
        description=(
            "Write content to a file in your workspace directory. "
            "Creates the file if it doesn't exist, overwrites if it does. "
            "Parent directories are created automatically. "
            "Path is relative to workspace root. Cannot write outside workspace."
        ),
        fn=file_write,
        parameters=[
            ToolParameter("path", "string", "File path relative to workspace (e.g. 'notes.md')"),
            ToolParameter("content", "string", "Content to write to the file"),
            ToolParameter(
                "append", "boolean",
                "If true, append to file instead of overwriting (default: false)",
                required=False,
            ),
        ],
        category="filesystem",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="file_list",
        description=(
            "List files in your workspace directory. "
            "Shows file names, sizes, and modification times. "
            "Path is relative to workspace root. Use '' or '.' for the root."
        ),
        fn=file_list,
        parameters=[
            ToolParameter("path", "string", "Directory path relative to workspace (default: root)", required=False),
            ToolParameter("recursive", "boolean", "List files recursively (default: false)", required=False),
        ],
        category="filesystem",
        source="shared",
        file_path=__file__,
    )


def _resolve_workspace() -> Path:
    """
    Get the current agent's workspace directory.

    Uses the ALFRED_WORKSPACE env var set by the agent runtime.
    Falls back to PROJECT_ROOT/workspaces/default if not set.
    """
    ws = os.environ.get("ALFRED_WORKSPACE", "")
    if ws:
        return Path(ws).resolve()

    from core.config import config
    return (config.PROJECT_ROOT / "workspaces" / "default").resolve()


def _safe_path(workspace: Path, user_path: str) -> Path:
    """
    Resolve a user path safely within the workspace sandbox.
    Raises ValueError if the path escapes the workspace.
    """
    # Normalize and resolve (workspace is already resolved)
    resolved = (workspace / user_path).resolve()

    # Check containment — use Path.is_relative_to for robust check
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise ValueError(f"Access denied: path '{user_path}' is outside your workspace.")

    return resolved


def file_read(path: str, max_chars: int = 20000) -> str:
    """Read a file from the agent's workspace."""
    max_chars = min(max(100, max_chars), 50000)

    workspace = _resolve_workspace()
    try:
        target = _safe_path(workspace, path)
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"File not found: {path}"

    if not target.is_file():
        return f"Not a file: {path} (use file_list to browse directories)"

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {path}: {e}"

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n... (truncated, {len(content)} chars total)"

    return content


def file_write(path: str, content: str, append: bool = False) -> str:
    """Write content to a file in the agent's workspace."""
    workspace = _resolve_workspace()
    try:
        target = _safe_path(workspace, path)
    except ValueError as e:
        return str(e)

    # Block writing to special files
    protected = {"SOUL.md", "AGENTS.md"}
    if target.name in protected:
        return f"Cannot write to {target.name} — this is a protected workspace file."

    try:
        # Create parent directories
        target.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(content)

        action = "Appended to" if append else "Wrote"
        return f"{action} {path} ({len(content)} chars)"

    except Exception as e:
        return f"Error writing {path}: {e}"


def file_list(path: str = ".", recursive: bool = False) -> str:
    """List files in a workspace directory."""
    workspace = _resolve_workspace()
    try:
        target = _safe_path(workspace, path or ".")
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"Directory not found: {path}"

    if not target.is_dir():
        return f"Not a directory: {path}"

    try:
        entries = []
        if recursive:
            for item in sorted(target.rglob("*")):
                if item.is_file():
                    rel = item.relative_to(workspace)
                    size = _human_size(item.stat().st_size)
                    entries.append(f"  {rel}  ({size})")
        else:
            for item in sorted(target.iterdir()):
                if item.name.startswith(".") and item.name not in (".env",):
                    continue
                rel = item.relative_to(workspace)
                if item.is_dir():
                    entries.append(f"  {rel}/")
                else:
                    size = _human_size(item.stat().st_size)
                    entries.append(f"  {rel}  ({size})")

        if not entries:
            return f"Directory '{path}' is empty."

        header = f"Files in {path or 'workspace root'}:"
        return header + "\n" + "\n".join(entries)

    except Exception as e:
        return f"Error listing {path}: {e}"


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
