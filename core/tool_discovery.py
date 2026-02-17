"""
Alfred AI - Tool Discovery
Auto-discover tools from directory conventions.

Tool sources:
  - shared:    tools/*.py       → Available to all agents
  - workspace: workspaces/X/tools/*.py → Only available to that agent

Convention: each .py file in a tools directory exports a register(registry) function.
The register() function calls registry.register_function() to add one or more tools.

Example tool file (tools/web_search.py):

    from core.tools import ToolRegistry, ToolParameter

    def register(registry: ToolRegistry):
        registry.register_function(
            name="web_search",
            description="Search the web",
            fn=web_search,
            parameters=[ToolParameter("query", "string", "Search query")],
            category="search",
            source="shared",
            file_path=__file__,
        )

    def web_search(query: str) -> str:
        ...
"""

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

from .tools import ToolRegistry
from .config import config


def discover_shared_tools(registry: ToolRegistry) -> list[str]:
    """
    Scan tools/*.py in the project root, call register(registry) on each.

    Returns:
        List of tool file names that were loaded.
    """
    tools_dir = config.PROJECT_ROOT / "tools"
    return _discover_tools_from_dir(registry, tools_dir, source="shared")


def discover_workspace_tools(registry: ToolRegistry, workspace_path: str) -> list[str]:
    """
    Scan workspaces/<name>/tools/*.py, call register(registry) on each.

    Returns:
        List of tool file names that were loaded.
    """
    tools_dir = Path(workspace_path) / "tools"
    return _discover_tools_from_dir(registry, tools_dir, source="workspace")


def _discover_tools_from_dir(registry: ToolRegistry, tools_dir: Path, source: str) -> list[str]:
    """
    Generic tool discovery from a directory.

    Scans for .py files (excluding __init__.py and __pycache__),
    loads each as a module, and calls its register(registry) function.
    """
    if not tools_dir.is_dir():
        return []

    loaded = []
    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue

        try:
            # Load the module dynamically
            module_name = f"alfred_tool_{source}_{py_file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, str(py_file))
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Call register(registry) if it exists
            if hasattr(module, "register") and callable(module.register):
                module.register(registry)
                loaded.append(py_file.name)
            else:
                print(f"  [discovery] Warning: {py_file.name} has no register() function, skipping")

        except Exception as e:
            print(f"  [discovery] Error loading {py_file.name}: {e}")
            traceback.print_exc()

    return loaded


def wrap_script_as_tool(
    script_path: str,
    name: str,
    description: str,
    category: str = "",
    source: str = "workspace",
) -> "Tool":
    """
    Wrap a standalone .py script as a Tool.

    The script is executed via subprocess with JSON input on stdin
    and is expected to print JSON output to stdout.

    Args:
        script_path: Path to the .py script
        name: Tool name
        description: Tool description
        category: Tool category
        source: Tool source tag

    Returns:
        A Tool instance that executes the script.
    """
    import subprocess
    import json
    from .tools import Tool, ToolParameter

    def execute(**kwargs) -> str:
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                input=json.dumps(kwargs),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return f"Script error: {result.stderr.strip()}"
            return result.stdout.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Script timed out after 60s"
        except Exception as e:
            return f"Error running script: {e}"

    return Tool(
        name=name,
        description=description,
        parameters=[
            ToolParameter("input", "string", "JSON input for the script", required=False),
        ],
        execute=execute,
        category=category,
        source=source,
        file_path=script_path,
    )
