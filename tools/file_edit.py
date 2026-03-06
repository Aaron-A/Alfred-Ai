"""
File Edit Tool
Surgically edit files with str_replace, insert at line, and delete operations.

Like OpenHands' OHEditor — agents can modify specific parts of a file without
rewriting the whole thing. Supports both workspace-scoped and project-scoped access.
"""

import os
import re
from pathlib import Path
from core.tools import ToolRegistry, ToolParameter
from core.config import config
from core.logging import get_logger

logger = get_logger("file_edit")

TOOL_META = {
    "version": "0.1.0",
    "author": "Alfred AI",
    "description": "Surgical file editing with str_replace, insert, and delete operations",
    "dependencies": [],
}


# ─── Path Resolution ──────────────────────────────────────────

def _resolve_path(file_path: str) -> Path:
    """
    Resolve a file path. Supports:
    - Relative paths (resolved against workspace)
    - Absolute paths within the project root
    - Paths starting with ~ (home dir)

    Blocks access to sensitive system directories.
    """
    # Expand ~ to home
    if file_path.startswith("~"):
        resolved = Path(file_path).expanduser().resolve()
    elif os.path.isabs(file_path):
        resolved = Path(file_path).resolve()
    else:
        # Relative to workspace
        ws = os.environ.get("ALFRED_WORKSPACE", "")
        if ws:
            resolved = (Path(ws) / file_path).resolve()
        else:
            resolved = (config.PROJECT_ROOT / file_path).resolve()

    # Safety: block sensitive directories
    blocked = ["/etc/shadow", "/etc/passwd", "/var/", "/usr/", "/bin/", "/sbin/"]
    path_str = str(resolved)
    for b in blocked:
        if path_str.startswith(b):
            raise ValueError(f"Access denied: cannot access {b}")

    return resolved


# ─── Core Functions ───────────────────────────────────────────

def file_view(file_path: str, start_line: int = None, end_line: int = None) -> str:
    """View a file or specific line range with line numbers."""
    try:
        target = _resolve_path(file_path)
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"File not found: {file_path}"
    if not target.is_file():
        return f"Not a file: {file_path}"

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {file_path}: {e}"

    lines = content.splitlines()
    total = len(lines)

    # Apply line range
    if start_line is not None or end_line is not None:
        start = max(1, start_line or 1)
        end = min(total, end_line or total)
        if start > total:
            return f"File has {total} lines, start_line {start} is beyond end."
        selected = lines[start - 1:end]
        header = f"File: {file_path} (lines {start}-{end} of {total})\n"
        numbered = [f"{i:>5}│ {line}" for i, line in enumerate(selected, start=start)]
    else:
        # Show whole file (cap at 500 lines)
        if total > 500:
            selected = lines[:500]
            header = f"File: {file_path} ({total} lines, showing first 500)\n"
        else:
            selected = lines
            header = f"File: {file_path} ({total} lines)\n"
        numbered = [f"{i:>5}│ {line}" for i, line in enumerate(selected, start=1)]

    return header + "\n".join(numbered)


def file_str_replace(file_path: str, old_string: str, new_string: str, count: int = 1) -> str:
    """Replace a string in a file. Shows a diff of the change."""
    try:
        target = _resolve_path(file_path)
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"File not found: {file_path}"
    if not target.is_file():
        return f"Not a file: {file_path}"

    # Block protected files
    protected = {"SOUL.md", "AGENTS.md"}
    if target.name in protected:
        return f"Cannot edit {target.name} — this is a protected workspace file."

    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {file_path}: {e}"

    # Check that old_string exists
    occurrences = content.count(old_string)
    if occurrences == 0:
        # Try to help the agent find the right string
        lines = content.splitlines()
        # Search for partial match
        first_line = old_string.strip().splitlines()[0] if old_string.strip() else ""
        matches = []
        if first_line:
            for i, line in enumerate(lines, 1):
                if first_line.strip() in line:
                    matches.append(f"  Line {i}: {line.rstrip()[:100]}")
        hint = ""
        if matches:
            hint = "\nPartial matches found:\n" + "\n".join(matches[:5])
        return f"String not found in {file_path}.{hint}"

    if count == 0:
        # Replace all
        new_content = content.replace(old_string, new_string)
        actual_count = occurrences
    else:
        new_content = content.replace(old_string, new_string, count)
        actual_count = min(count, occurrences)

    target.write_text(new_content, encoding="utf-8")

    # Generate compact diff
    old_preview = old_string[:100] + ("..." if len(old_string) > 100 else "")
    new_preview = new_string[:100] + ("..." if len(new_string) > 100 else "")
    return (
        f"Replaced {actual_count} occurrence(s) in {file_path}\n"
        f"  - {old_preview}\n"
        f"  + {new_preview}"
    )


def file_insert(file_path: str, line_number: int, content: str) -> str:
    """Insert content at a specific line number (1-indexed). Existing content shifts down."""
    try:
        target = _resolve_path(file_path)
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"File not found: {file_path}"

    protected = {"SOUL.md", "AGENTS.md"}
    if target.name in protected:
        return f"Cannot edit {target.name} — this is a protected workspace file."

    try:
        existing = target.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {file_path}: {e}"

    lines = existing.splitlines(keepends=True)
    # Clamp line_number
    line_number = max(1, min(line_number, len(lines) + 1))

    # Insert the new content
    new_lines = content.splitlines(keepends=True)
    if not content.endswith("\n"):
        new_lines.append("\n")

    lines[line_number - 1:line_number - 1] = new_lines
    target.write_text("".join(lines), encoding="utf-8")

    inserted_count = len(new_lines)
    return (
        f"Inserted {inserted_count} line(s) at line {line_number} in {file_path}\n"
        f"  File now has {len(lines)} lines."
    )


def file_delete(file_path: str) -> str:
    """Delete a file or empty directory."""
    try:
        target = _resolve_path(file_path)
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"Not found: {file_path}"

    protected = {"SOUL.md", "AGENTS.md", "TOOLS.md", "USER.md", "alfred.json"}
    if target.name in protected:
        return f"Cannot delete {target.name} — this is a protected file."

    try:
        if target.is_file():
            size = target.stat().st_size
            target.unlink()
            return f"Deleted file: {file_path} ({size} bytes)"
        elif target.is_dir():
            if any(target.iterdir()):
                return f"Cannot delete {file_path} — directory is not empty. Use code_exec with bash to force-delete."
            target.rmdir()
            return f"Deleted empty directory: {file_path}"
        else:
            return f"Cannot delete {file_path} — unknown type."
    except Exception as e:
        return f"Error deleting {file_path}: {e}"


# ─── Tool Registration ──────────────────────────────────────────

def register(registry: ToolRegistry):
    """Register file editing tools."""

    registry.register_function(
        name="file_view",
        description=(
            "View a file's contents with line numbers. Supports viewing specific "
            "line ranges. Accepts relative paths (workspace), absolute paths, or ~ paths. "
            "Use this before editing to see the current state of a file."
        ),
        fn=file_view,
        parameters=[
            ToolParameter("file_path", "string",
                "File path — relative to workspace, absolute, or ~/..."),
            ToolParameter("start_line", "integer",
                "Start line number (1-indexed, optional)", required=False),
            ToolParameter("end_line", "integer",
                "End line number (inclusive, optional)", required=False),
        ],
        category="filesystem",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="file_str_replace",
        description=(
            "Replace a specific string in a file. The old_string must match EXACTLY "
            "(including whitespace and indentation). Use file_view first to see the "
            "exact content. Set count=0 to replace ALL occurrences. "
            "Shows a diff of what changed."
        ),
        fn=file_str_replace,
        parameters=[
            ToolParameter("file_path", "string",
                "File path — relative to workspace, absolute, or ~/..."),
            ToolParameter("old_string", "string",
                "Exact string to find and replace (must match exactly, including whitespace)"),
            ToolParameter("new_string", "string",
                "Replacement string"),
            ToolParameter("count", "integer",
                "Number of occurrences to replace (default: 1, use 0 for all)",
                required=False),
        ],
        category="filesystem",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="file_insert",
        description=(
            "Insert content at a specific line number in a file. "
            "Existing content shifts down. Line numbers are 1-indexed. "
            "Use file_view first to see line numbers."
        ),
        fn=file_insert,
        parameters=[
            ToolParameter("file_path", "string",
                "File path — relative to workspace, absolute, or ~/..."),
            ToolParameter("line_number", "integer",
                "Line number to insert at (1-indexed, content shifts down)"),
            ToolParameter("content", "string",
                "Content to insert (can be multiple lines)"),
        ],
        category="filesystem",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="file_delete",
        description=(
            "Delete a file or empty directory. Protected workspace files "
            "(SOUL.md, AGENTS.md, alfred.json) cannot be deleted."
        ),
        fn=file_delete,
        parameters=[
            ToolParameter("file_path", "string",
                "File path — relative to workspace, absolute, or ~/..."),
        ],
        category="filesystem",
        source="shared",
        file_path=__file__,
    )
