"""
Alfred AI - Meta-Tools
Tools that agents use to manage their own toolset.

These let an agent:
  - List what tools it has
  - Search for existing tools before creating duplicates
  - Create new workspace tools on-the-fly
  - Remove workspace tools it no longer needs
"""

import os
import textwrap
from pathlib import Path

from .tools import ToolRegistry, ToolParameter


def register_meta_tools(registry: ToolRegistry, agent_name: str, workspace_path: str):
    """
    Register meta-tools that let an agent manage its own tools.

    Args:
        registry: The agent's tool registry
        agent_name: The agent's name (for scoping)
        workspace_path: Path to agent's workspace (for tool file creation)
    """
    tools_dir = Path(workspace_path) / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    # ─── tool_list ─────────────────────────────────────────────
    def tool_list() -> str:
        """List all tools available to this agent."""
        return registry.to_manifest()

    registry.register_function(
        name="tool_list",
        description=(
            "List all tools available to you. Shows name, description, "
            "category, source, and parameters for every tool in your registry. "
            "Call this to understand what capabilities you have."
        ),
        fn=tool_list,
        parameters=[],
        category="meta",
        source="builtin",
    )

    # ─── tool_search ───────────────────────────────────────────
    def tool_search(query: str) -> str:
        """Search for existing tools by name or description."""
        query_lower = query.lower()
        matches = []
        for tool in registry.all():
            # Search in name and description
            if (query_lower in tool.name.lower()
                    or query_lower in tool.description.lower()
                    or query_lower in tool.category.lower()):
                source_tag = f" [{tool.source}]" if tool.source != "builtin" else ""
                matches.append(
                    f"- {tool.name}{source_tag} ({tool.category or 'general'}): "
                    f"{tool.description[:120]}"
                )

        if not matches:
            return f"No tools found matching '{query}'. You can create one with tool_create."
        return f"Found {len(matches)} matching tool(s):\n" + "\n".join(matches)

    registry.register_function(
        name="tool_search",
        description=(
            "Check if a tool with a given name or capability already exists. "
            "ALWAYS call this before creating a new tool to avoid duplicates. "
            "Searches tool names, descriptions, and categories."
        ),
        fn=tool_search,
        parameters=[
            ToolParameter("query", "string", "Search term — tool name, keyword, or capability"),
        ],
        category="meta",
        source="builtin",
    )

    # ─── tool_create ───────────────────────────────────────────
    def tool_create(
        name: str,
        description: str,
        code: str,
        category: str = "custom",
        dependencies: str = "",
    ) -> str:
        """Create a new tool in the agent's workspace."""
        # Validate name
        if not name.isidentifier():
            return f"Error: Invalid tool name '{name}'. Must be a valid Python identifier (letters, numbers, underscores)."

        # Check for duplicates
        if registry.has_tool(name):
            existing = registry.get(name)
            return (
                f"Error: Tool '{name}' already exists (source: {existing.source}, "
                f"category: {existing.category}). Use a different name or remove it first."
            )

        # Write the tool file
        tool_file = tools_dir / f"{name}.py"

        # Build the tool file content
        dep_list = [d.strip() for d in dependencies.split(",") if d.strip()] if dependencies else []
        dep_str = repr(dep_list) if dep_list else "[]"

        file_content = textwrap.dedent(f'''\
            """Auto-generated tool: {name}"""
            from core.tools import ToolRegistry, ToolParameter

            def register(registry: ToolRegistry):
                registry.register_function(
                    name="{name}",
                    description="""{description}""",
                    fn={name},
                    parameters=_get_parameters(),
                    category="{category}",
                    source="workspace",
                    file_path=__file__,
                    dependencies={dep_str},
                )

            def _get_parameters():
                """Define parameters by inspecting the function signature."""
                import inspect
                params = []
                sig = inspect.signature({name})
                for pname, param in sig.parameters.items():
                    ptype = "string"  # default
                    if param.annotation != inspect.Parameter.empty:
                        ann = param.annotation
                        if ann in (int,):
                            ptype = "integer"
                        elif ann in (float,):
                            ptype = "number"
                        elif ann in (bool,):
                            ptype = "boolean"
                    required = param.default is inspect.Parameter.empty
                    desc = pname.replace("_", " ")
                    params.append(ToolParameter(pname, ptype, desc, required=required))
                return params

        ''') + code + "\n"

        try:
            tool_file.write_text(file_content)
        except Exception as e:
            return f"Error writing tool file: {e}"

        # Install dependencies if needed
        if dep_list:
            import subprocess
            import sys
            for dep in dep_list:
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", dep],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                except Exception as e:
                    return f"Tool file created but dependency '{dep}' failed to install: {e}"

        # Register the new tool immediately
        try:
            from .tool_discovery import discover_workspace_tools
            discover_workspace_tools(registry, workspace_path)
        except Exception as e:
            return f"Tool file created at {tool_file} but failed to register: {e}"

        if registry.has_tool(name):
            return (
                f"Tool '{name}' created and registered successfully.\n"
                f"  File: {tool_file}\n"
                f"  Category: {category}\n"
                f"  Source: workspace"
            )
        else:
            return f"Tool file created at {tool_file} but registration failed. Check the code for errors."

    registry.register_function(
        name="tool_create",
        description=(
            "Create a new tool in your workspace. Writes a Python file and registers it immediately. "
            "The 'code' parameter should contain ONLY the function definition (def function_name(...): ...). "
            "The boilerplate (imports, register(), parameter detection) is generated automatically. "
            "REQUIREMENTS: "
            "1. Always call tool_search first to check if the tool already exists. "
            "2. The function name must match the tool name exactly (snake_case). "
            "3. Use type hints on parameters (str, int, float, bool) — they are auto-detected. "
            "4. The function must return a string. "
            "5. Wrap external calls in try/except and return error messages as strings. "
            "6. List any pip packages in the dependencies field."
        ),
        fn=tool_create,
        parameters=[
            ToolParameter("name", "string", "Tool name in snake_case (e.g. 'get_price', 'parse_csv')"),
            ToolParameter("description", "string", "Clear description of what this tool does — this is shown to the LLM when deciding which tool to call"),
            ToolParameter("code", "string", "The Python function code. Must define a function with the same name as the tool. Use type hints for parameters. Must return a string."),
            ToolParameter("category", "string", "Tool category (e.g. 'data', 'search', 'web', 'custom')", required=False),
            ToolParameter("dependencies", "string", "Comma-separated pip packages needed (e.g. 'requests,beautifulsoup4')", required=False),
        ],
        category="meta",
        source="builtin",
    )

    # ─── tool_remove ───────────────────────────────────────────
    def tool_remove(name: str) -> str:
        """Remove a workspace tool."""
        tool = registry.get(name)
        if tool is None:
            return f"Error: Tool '{name}' not found."

        if tool.source != "workspace":
            return (
                f"Error: Cannot remove '{name}' — it's a {tool.source} tool. "
                f"Only workspace tools can be removed."
            )

        # Unregister from registry
        registry.unregister(name)

        # Delete the file
        if tool.file_path:
            try:
                file_path = Path(tool.file_path)
                if file_path.exists():
                    file_path.unlink()
                    return f"Tool '{name}' removed. File deleted: {file_path}"
                else:
                    return f"Tool '{name}' unregistered but file not found: {file_path}"
            except Exception as e:
                return f"Tool '{name}' unregistered but file deletion failed: {e}"

        return f"Tool '{name}' unregistered (no file path recorded)."

    registry.register_function(
        name="tool_remove",
        description=(
            "Remove a workspace tool you created. Unregisters it and deletes the file. "
            "Can only remove tools with source='workspace' — builtin and shared tools cannot be removed."
        ),
        fn=tool_remove,
        parameters=[
            ToolParameter("name", "string", "Name of the workspace tool to remove"),
        ],
        category="meta",
        source="builtin",
    )
