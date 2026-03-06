"""
Grep/Search Tool
Search file contents across directories using regex or literal patterns.

Like ripgrep — agents can find code, text, or patterns across an entire project
without reading every file manually.
"""

import os
import re
import fnmatch
from pathlib import Path
from core.tools import ToolRegistry, ToolParameter
from core.config import config
from core.logging import get_logger

logger = get_logger("grep_search")

TOOL_META = {
    "version": "0.1.0",
    "author": "Alfred AI",
    "description": "Search file contents with regex patterns across directories",
    "dependencies": [],
}

# Max results to prevent output explosion
MAX_RESULTS = 100
MAX_LINE_LEN = 300

# Directories always skipped
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".tox", "dist", "build", ".egg-info", ".browser_state",
    ".code_exec", "screenshots",
}

# Binary file extensions to skip
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".ogg", ".webm", ".avi",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt",
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".sqlite", ".db", ".whl",
}


def _resolve_search_path(search_path: str) -> Path:
    """Resolve search path, similar to file_edit's _resolve_path."""
    if search_path.startswith("~"):
        return Path(search_path).expanduser().resolve()
    elif os.path.isabs(search_path):
        return Path(search_path).resolve()
    else:
        ws = os.environ.get("ALFRED_WORKSPACE", "")
        if ws:
            return (Path(ws) / search_path).resolve()
        return (config.PROJECT_ROOT / search_path).resolve()


def _should_skip_file(path: Path, include_glob: str = None) -> bool:
    """Check if a file should be skipped."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    if include_glob and not fnmatch.fnmatch(path.name, include_glob):
        return True
    return False


def grep(
    pattern: str,
    path: str = ".",
    include: str = None,
    ignore_case: bool = False,
    max_results: int = 50,
    context_lines: int = 0,
) -> str:
    """Search file contents for a pattern."""
    try:
        search_root = _resolve_search_path(path)
    except Exception as e:
        return f"Error resolving path: {e}"

    if not search_root.exists():
        return f"Path not found: {path}"

    max_results = min(max(1, max_results), MAX_RESULTS)
    context_lines = min(max(0, context_lines), 5)

    # Compile regex
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    results = []
    files_searched = 0
    files_matched = 0

    # Search a single file
    if search_root.is_file():
        files = [search_root]
    else:
        # Walk directory
        files = []
        for dirpath, dirnames, filenames in os.walk(search_root):
            # Skip excluded directories (in-place mutation for os.walk)
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                if not _should_skip_file(fpath, include):
                    files.append(fpath)

    for fpath in files:
        files_searched += 1
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = content.splitlines()
        file_matches = []

        for i, line in enumerate(lines):
            if regex.search(line):
                match_entry = {"line_num": i + 1, "line": line}

                # Add context lines
                if context_lines > 0:
                    ctx_before = []
                    ctx_after = []
                    for ci in range(max(0, i - context_lines), i):
                        ctx_before.append((ci + 1, lines[ci]))
                    for ci in range(i + 1, min(len(lines), i + 1 + context_lines)):
                        ctx_after.append((ci + 1, lines[ci]))
                    match_entry["before"] = ctx_before
                    match_entry["after"] = ctx_after

                file_matches.append(match_entry)

        if file_matches:
            files_matched += 1
            # Relative path for cleaner output
            try:
                rel = fpath.relative_to(search_root)
            except ValueError:
                rel = fpath
            results.append({"file": str(rel), "matches": file_matches})

        if sum(len(r["matches"]) for r in results) >= max_results:
            break

    if not results:
        return f"No matches for '{pattern}' in {path} ({files_searched} files searched)"

    # Format output
    total_matches = sum(len(r["matches"]) for r in results)
    output = [f"Found {total_matches} matches in {files_matched} files ({files_searched} searched)\n"]

    for r in results:
        output.append(f"── {r['file']} ──")
        for m in r["matches"]:
            line_text = m["line"][:MAX_LINE_LEN]
            if len(m["line"]) > MAX_LINE_LEN:
                line_text += "..."

            if context_lines > 0 and ("before" in m or "after" in m):
                for ln, lt in m.get("before", []):
                    output.append(f"  {ln:>5}  {lt[:MAX_LINE_LEN]}")
                output.append(f"  {m['line_num']:>5}▸ {line_text}")
                for ln, lt in m.get("after", []):
                    output.append(f"  {ln:>5}  {lt[:MAX_LINE_LEN]}")
                output.append("")
            else:
                output.append(f"  {m['line_num']:>5}│ {line_text}")
        output.append("")

    return "\n".join(output)


def find_files(
    pattern: str,
    path: str = ".",
    file_type: str = None,
    max_results: int = 50,
) -> str:
    """Find files by name pattern (glob). Optionally filter by type."""
    try:
        search_root = _resolve_search_path(path)
    except Exception as e:
        return f"Error resolving path: {e}"

    if not search_root.exists():
        return f"Path not found: {path}"

    max_results = min(max(1, max_results), 200)

    results = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in sorted(filenames):
            if fnmatch.fnmatch(fname, pattern):
                fpath = Path(dirpath) / fname

                # Filter by type
                if file_type == "file" and not fpath.is_file():
                    continue
                if file_type == "dir" and not fpath.is_dir():
                    continue

                try:
                    rel = fpath.relative_to(search_root)
                except ValueError:
                    rel = fpath

                size = fpath.stat().st_size
                results.append((str(rel), size))

                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break

    if not results:
        return f"No files matching '{pattern}' in {path}"

    output = [f"Found {len(results)} files matching '{pattern}':\n"]
    for rel_path, size in results:
        if size < 1024:
            sz = f"{size} B"
        elif size < 1024 * 1024:
            sz = f"{size / 1024:.1f} KB"
        else:
            sz = f"{size / (1024 * 1024):.1f} MB"
        output.append(f"  {rel_path}  ({sz})")

    return "\n".join(output)


# ─── Tool Registration ──────────────────────────────────────────

def register(registry: ToolRegistry):
    """Register grep/search tools."""

    registry.register_function(
        name="grep",
        description=(
            "Search file contents for a regex pattern across a directory tree. "
            "Like ripgrep — finds matching lines with file paths and line numbers. "
            "Skips binary files, .git, node_modules, .venv automatically. "
            "Accepts relative paths (workspace), absolute paths, or ~ paths."
        ),
        fn=grep,
        parameters=[
            ToolParameter("pattern", "string",
                "Regex pattern to search for (e.g. 'def main', 'TODO', 'import.*json')"),
            ToolParameter("path", "string",
                "Directory or file to search in (default: workspace root)",
                required=False),
            ToolParameter("include", "string",
                "Glob pattern to filter files (e.g. '*.py', '*.json')",
                required=False),
            ToolParameter("ignore_case", "boolean",
                "Case-insensitive search (default: false)",
                required=False),
            ToolParameter("max_results", "integer",
                "Max matches to return (default: 50, max: 100)",
                required=False),
            ToolParameter("context_lines", "integer",
                "Lines of context before/after each match (0-5, default: 0)",
                required=False),
        ],
        category="filesystem",
        source="shared",
        file_path=__file__,
    )

    registry.register_function(
        name="find_files",
        description=(
            "Find files by name pattern (glob). Searches recursively through "
            "directories, skipping .git, node_modules, .venv. "
            "Returns file paths and sizes."
        ),
        fn=find_files,
        parameters=[
            ToolParameter("pattern", "string",
                "Glob pattern for filenames (e.g. '*.py', 'config.*', '*.json')"),
            ToolParameter("path", "string",
                "Directory to search in (default: workspace root)",
                required=False),
            ToolParameter("file_type", "string",
                "Filter by type: 'file' or 'dir' (default: all)",
                required=False,
                enum=["file", "dir"]),
            ToolParameter("max_results", "integer",
                "Max results to return (default: 50, max: 200)",
                required=False),
        ],
        category="filesystem",
        source="shared",
        file_path=__file__,
    )
