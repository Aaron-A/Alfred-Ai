"""
Alfred AI - Tool Registry
Define tools once, grant them to any agent.

Tools are Python functions that agents can call. Each tool has:
- A name and description (for the LLM to understand what it does)
- A parameter schema (so the LLM knows what arguments to pass)
- An execute function (the actual implementation)

The registry makes tools available to agents without copying code around.
"""

import json
import inspect
from typing import Callable, Any
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    """A single parameter for a tool."""
    name: str
    type: str  # "string", "number", "boolean", "integer"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] = None


@dataclass
class Tool:
    """A tool that agents can call."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    execute: Callable = None
    category: str = ""  # e.g., "trading", "social", "search", "memory"
    source: str = "builtin"  # "builtin", "shared", "workspace"
    file_path: str = ""  # Path to the .py file that defines this tool
    dependencies: list[str] = field(default_factory=list)  # pip packages needed

    def to_schema(self) -> dict:
        """Convert to LLM-compatible tool schema (OpenAI function calling format)."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic tool_use format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def run(self, **kwargs) -> str:
        """Execute the tool with given arguments. Returns string result."""
        if self.execute is None:
            return f"Error: Tool '{self.name}' has no execute function"
        try:
            result = self.execute(**kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return f"Error executing {self.name}: {e}"


class ToolRegistry:
    """
    Central registry of all available tools.

    Register tools once, then grant subsets to different agents.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        fn: Callable,
        parameters: list[ToolParameter] = None,
        category: str = "",
        source: str = "builtin",
        file_path: str = "",
        dependencies: list[str] = None,
    ) -> Tool:
        """Register a plain function as a tool (convenience method)."""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters or [],
            execute=fn,
            category=category,
            source=source,
            file_path=file_path,
            dependencies=dependencies or [],
        )
        self.register(tool)
        return tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_many(self, names: list[str]) -> list[Tool]:
        """Get multiple tools by name."""
        return [self._tools[n] for n in names if n in self._tools]

    def get_by_category(self, category: str) -> list[Tool]:
        """Get all tools in a category."""
        return [t for t in self._tools.values() if t.category == category]

    def all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def schemas(self, names: list[str] = None, format: str = "openai") -> list[dict]:
        """
        Get tool schemas for LLM consumption.

        Args:
            names: Specific tool names (None = all tools)
            format: "openai" or "anthropic"
        """
        tools = self.get_many(names) if names else self.all()
        if format == "anthropic":
            return [t.to_anthropic_schema() for t in tools]
        return [t.to_schema() for t in tools]

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name with given arguments."""
        tool = self.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'"
        return tool.run(**arguments)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_by_source(self, source: str) -> list[Tool]:
        """Get all tools from a specific source (builtin, shared, workspace)."""
        return [t for t in self._tools.values() if t.source == source]

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if found and removed."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def to_manifest(self, names: list[str] = None) -> str:
        """
        Generate a human-readable manifest of tools.

        Args:
            names: Specific tool names (None = all)

        Returns:
            Formatted string listing all tools with descriptions and parameters.
        """
        tools = self.get_many(names) if names else self.all()
        if not tools:
            return "No tools available."

        # Group by category
        categories: dict[str, list[Tool]] = {}
        for tool in tools:
            cat = tool.category or "general"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)

        lines = []
        for cat in sorted(categories.keys()):
            cat_tools = categories[cat]
            lines.append(f"## {cat.title()} Tools")
            lines.append("")
            for tool in cat_tools:
                source_tag = f" [{tool.source}]" if tool.source != "builtin" else ""
                lines.append(f"### {tool.name}{source_tag}")
                lines.append(tool.description)
                if tool.parameters:
                    for p in tool.parameters:
                        req = "required" if p.required else "optional"
                        lines.append(f"  - `{p.name}` ({p.type}, {req}): {p.description}")
                lines.append("")

        return "\n".join(lines)


# ─── Singleton ───────────────────────────────────────────────────

_registry = None

def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


# ─── Built-in Tools ─────────────────────────────────────────────

def register_builtin_tools(registry: ToolRegistry):
    """Register Alfred's built-in tools."""

    # Memory search tool (agents can search their own memories)
    # Note: imports are deferred to execution time so tool registration
    # doesn't fail if heavy dependencies (lancedb) aren't installed
    def memory_search(query: str, memory_type: str = None, top_k: int = 5) -> str:
        """Search agent memory for relevant context."""
        from .memory import MemoryStore
        store = MemoryStore()
        results = store.search(query=query, memory_type=memory_type or None, top_k=top_k)
        if not results:
            return "No relevant memories found."
        lines = []
        for r in results:
            mtype = r.get("memory_type", "?")
            content = r.get("content", "")
            dist = r.get("_distance", 0)
            relevance = max(0, 1 - dist)
            lines.append(f"[{mtype}] (relevance: {relevance:.0%}) {content[:200]}")
        return "\n".join(lines)

    registry.register_function(
        name="memory_search",
        description="Search your memory for relevant past events, trades, decisions, or context. Use this before answering questions about prior work, past trades, lessons learned, or historical context.",
        fn=memory_search,
        parameters=[
            ToolParameter("query", "string", "Natural language search query"),
            ToolParameter("memory_type", "string", "Filter to type: trade, macro, tweet, decision (optional)", required=False),
            ToolParameter("top_k", "integer", "Number of results (default 5)", required=False),
        ],
        category="memory",
    )

    # Memory store tool (agents can save new memories)
    def memory_store(content: str, memory_type: str = "generic", tags: str = "") -> str:
        """Store a new memory."""
        from .memory import MemoryStore
        from models.base import MemoryRecord
        store = MemoryStore()
        record = MemoryRecord(
            content=content,
            memory_type=memory_type,
            tags=tags,
        )
        record_id = store.store(record)
        return f"Memory stored (id: {record_id})"

    registry.register_function(
        name="memory_store",
        description="Save a new memory for future recall. Use this to remember important decisions, lessons learned, or context you'll need later.",
        fn=memory_store,
        parameters=[
            ToolParameter("content", "string", "What to remember"),
            ToolParameter("memory_type", "string", "Type: generic, trade, macro, tweet, decision", required=False),
            ToolParameter("tags", "string", "Comma-separated tags for filtering", required=False),
        ],
        category="memory",
    )

    # Shell command tool (controlled execution)
    import subprocess

    def run_command(command: str, timeout: int = 30) -> str:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(config.PROJECT_ROOT),
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    from .config import config

    registry.register_function(
        name="run_command",
        description="Execute a shell command. Use for running scripts, checking system state, or executing tools. Be careful with destructive commands.",
        fn=run_command,
        parameters=[
            ToolParameter("command", "string", "The shell command to execute"),
            ToolParameter("timeout", "integer", "Timeout in seconds (default 30)", required=False),
        ],
        category="system",
    )
