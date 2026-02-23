"""
Alfred AI - Community Tool Registry
Manages installation, scaffolding, and discovery of community-contributed tools.
"""

import json
import urllib.request
from pathlib import Path

from .config import config
from .logging import get_logger

logger = get_logger("tool_registry")

# Default registry index URL (GitHub-hosted JSON)
DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/Aaron-A/Alfred-Ai/main/registry/tools.json"
)


def get_tools_dir() -> Path:
    """Get the shared tools directory."""
    tools_dir = config.PROJECT_ROOT / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    return tools_dir


def fetch_registry_index(registry_url: str = None) -> dict:
    """Fetch the tool registry index from GitHub."""
    url = registry_url or DEFAULT_REGISTRY_URL
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Alfred-AI"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"Failed to fetch registry: {e}")
        return {"tools": []}


def install_tool_from_url(url: str, name: str = None) -> tuple[bool, str]:
    """
    Download a tool .py file from a URL into the tools directory.

    Returns:
        (success, message) tuple
    """
    tools_dir = get_tools_dir()

    # Determine filename
    if name:
        filename = f"{name}.py" if not name.endswith(".py") else name
    else:
        filename = url.rstrip("/").split("/")[-1]
        if not filename.endswith(".py"):
            filename += ".py"

    target = tools_dir / filename

    if target.exists():
        return False, f"Tool already exists: {target.name}. Remove it first with: alfred tool remove {target.stem}"

    # Download
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Alfred-AI"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode()
    except Exception as e:
        return False, f"Failed to download: {e}"

    # Validate: must have register() function
    if "def register(" not in content:
        return False, "Invalid tool file: no register() function found"

    # Write to disk
    target.write_text(content)
    return True, f"Installed: {target.name} -> {target}"


def install_tool_from_registry(tool_name: str) -> tuple[bool, str]:
    """Install a tool by name from the registry index."""
    index = fetch_registry_index()
    tools = {t["name"]: t for t in index.get("tools", [])}

    if tool_name not in tools:
        available = ", ".join(tools.keys()) if tools else "(empty registry)"
        return False, f"Tool '{tool_name}' not found in registry. Available: {available}"

    entry = tools[tool_name]
    url = entry.get("url", "")
    if not url:
        return False, f"Tool '{tool_name}' has no download URL in registry."

    return install_tool_from_url(url, name=tool_name)


def remove_tool(name: str) -> tuple[bool, str]:
    """Remove a community-installed tool file."""
    tools_dir = get_tools_dir()
    filename = f"{name}.py" if not name.endswith(".py") else name
    target = tools_dir / filename

    if not target.exists():
        return False, f"Tool file not found: {target}"

    target.unlink()
    return True, f"Removed: {target.name}"


def create_tool_scaffold(name: str, workspace: str = None) -> tuple[bool, str]:
    """
    Create a new tool file from a scaffold template.

    Args:
        name: Tool name (used as filename and function name)
        workspace: Optional workspace path for workspace-local tools
    """
    if workspace:
        target_dir = Path(workspace) / "tools"
    else:
        target_dir = get_tools_dir()

    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize name for Python identifier
    safe_name = name.replace("-", "_").replace(" ", "_")
    target = target_dir / f"{safe_name}.py"

    if target.exists():
        return False, f"File already exists: {target}"

    scaffold = f'''"""
Tool: {name}
Description: TODO — describe what this tool does
"""

from core.tools import ToolRegistry, ToolParameter

TOOL_META = {{
    "version": "0.1.0",
    "author": "",
    "description": "TODO: describe what this tool does",
    "dependencies": [],
}}


def register(registry: ToolRegistry):
    registry.register_function(
        name="{safe_name}",
        description="TODO: describe what this tool does",
        fn={safe_name},
        parameters=[
            ToolParameter("input", "string", "TODO: describe parameter"),
        ],
        category="custom",
        source="shared",
        file_path=__file__,
    )


def {safe_name}(input: str) -> str:
    """TODO: implement tool logic."""
    return f"{{input}}"
'''
    target.write_text(scaffold)
    return True, f"Created: {target}"
