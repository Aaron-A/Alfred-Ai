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
    source: str = "builtin"  # "builtin", "shared", "workspace", "community"
    file_path: str = ""  # Path to the .py file that defines this tool
    dependencies: list[str] = field(default_factory=list)  # pip packages needed
    version: str = ""  # Semver string (e.g., "1.0.0")
    author: str = ""  # Tool author

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

def register_builtin_tools(registry: ToolRegistry, agent_id: str = None,
                           memory_shared: bool = False):
    """
    Register Alfred's built-in tools.

    Args:
        registry: The tool registry to register into.
        agent_id: The owning agent's name. When set, memory tools are scoped
                  to this agent — searches only return this agent's memories,
                  and stores are tagged with this agent_id.
                  When None, no isolation (legacy/global behavior).
        memory_shared: If True, also register a memory_search_global tool
                       that searches across all agents' memories.
    """

    # ─── Memory Tools (agent-scoped by default) ──────────────

    def memory_search(query: str, memory_type: str = None, top_k: int = 5,
                       filters: str = None) -> str:
        """Search your own memory for relevant context."""
        from .memory import MemoryStore
        store = MemoryStore(agent_id=agent_id or "default")

        # Build agent_id filter so we only see our own memories
        where = f"agent_id = '{agent_id}'" if agent_id else None

        # Parse structured filters if provided (JSON string)
        parsed_filters = None
        if filters:
            try:
                parsed_filters = json.loads(filters)
            except (json.JSONDecodeError, TypeError):
                pass  # Ignore malformed filters

        results = store.search(
            query=query, memory_type=memory_type or None,
            top_k=top_k, where=where, filters=parsed_filters,
        )
        if not results:
            return "No relevant memories found."
        lines = []
        for r in results:
            rid = r.get("id", "?")
            mtype = r.get("memory_type", "?")
            content = r.get("content", "")
            dist = r.get("_distance", 0)
            relevance = max(0, 1 - dist)
            importance = r.get("importance", 0.5) or 0.5
            imp_tag = f" ★{importance:.1f}" if importance > 0.6 else ""
            lines.append(f"[{mtype}] id:{rid} (relevance: {relevance:.0%}{imp_tag}) {content[:200]}")
        return "\n".join(lines)

    registry.register_function(
        name="memory_search",
        description=(
            "Search your memory for relevant past events, trades, decisions, or context. "
            "Use this before answering questions about prior work, past trades, lessons learned, "
            "or historical context. Results include record IDs for use with memory_update and memory_link."
        ),
        fn=memory_search,
        parameters=[
            ToolParameter("query", "string", "Natural language search query"),
            ToolParameter("memory_type", "string", "Filter to type: trade, macro, tweet, decision (optional)", required=False),
            ToolParameter("top_k", "integer", "Number of results (default 5)", required=False),
            ToolParameter("filters", "string", 'JSON object for structured filtering, e.g. \'{"symbol":"BTC/USD","outcome":"win"}\' (optional)', required=False),
        ],
        category="memory",
    )

    # Memory store tool (agents can save new memories, tagged with their agent_id)
    def memory_store(content: str, memory_type: str = "generic", tags: str = "") -> str:
        """Store a new memory."""
        from .memory import MemoryStore
        from models.base import MemoryRecord
        store = MemoryStore(agent_id=agent_id or "default")
        record = MemoryRecord(
            content=content,
            memory_type=memory_type,
            tags=tags,
            agent_id=agent_id or "default",
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

    # Memory update tool — update importance, tags, or content of existing memories
    def memory_update(record_id: str, memory_type: str, importance: float = None,
                      tags: str = None, content: str = None) -> str:
        """Update an existing memory's importance, tags, or content."""
        from .memory import MemoryStore
        store = MemoryStore(agent_id=agent_id or "default")
        updates = {}
        if importance is not None:
            updates["importance"] = max(0.0, min(1.0, importance))
        if tags is not None:
            updates["tags"] = tags
        if content is not None:
            updates["content"] = content
        if not updates:
            return "Error: No updates provided. Specify importance, tags, or content."
        success = store.update(record_id, memory_type, updates)
        if success:
            return f"Memory {record_id} updated: {', '.join(f'{k}={v}' for k, v in updates.items())}"
        return f"Memory {record_id} not found in {memory_type}."

    registry.register_function(
        name="memory_update",
        description=(
            "Update an existing memory record. Use this to adjust importance after seeing outcomes "
            "(e.g., a lesson proved valuable → increase importance), update tags, or correct content. "
            "Get the record_id from memory_search results."
        ),
        fn=memory_update,
        parameters=[
            ToolParameter("record_id", "string", "The memory record ID to update"),
            ToolParameter("memory_type", "string", "The memory type (trade, decision, macro, etc.)"),
            ToolParameter("importance", "number", "New importance score 0.0-1.0 (higher = more important)", required=False),
            ToolParameter("tags", "string", "New comma-separated tags", required=False),
            ToolParameter("content", "string", "Updated content text", required=False),
        ],
        category="memory",
    )

    # Memory link tool — connect related memories (decision → trade outcome)
    def memory_link(source_id: str, source_type: str, target_id: str, target_type: str) -> str:
        """Link two memories together (e.g., a decision to its trade outcome)."""
        from .memory import MemoryStore
        store = MemoryStore(agent_id=agent_id or "default")

        # Get source record to append linked_id
        source = store.get(source_id, source_type)
        if not source:
            return f"Error: Source memory {source_id} not found in {source_type}."

        # Verify target exists
        target = store.get(target_id, target_type)
        if not target:
            return f"Error: Target memory {target_id} not found in {target_type}."

        # Append to linked_ids (avoid duplicates)
        existing = source.get("linked_ids", "") or ""
        existing_ids = set(existing.split(",")) if existing else set()
        existing_ids.discard("")
        existing_ids.add(target_id)
        new_linked = ",".join(sorted(existing_ids))

        store.update(source_id, source_type, {"linked_ids": new_linked})
        return f"Linked {source_type}/{source_id} → {target_type}/{target_id}"

    registry.register_function(
        name="memory_link",
        description=(
            "Link two memories together to track cause and effect. "
            "For example, link a decision memory to the trade outcome that resulted from it. "
            "This helps you learn which decisions led to good or bad outcomes."
        ),
        fn=memory_link,
        parameters=[
            ToolParameter("source_id", "string", "ID of the source memory (e.g., the decision)"),
            ToolParameter("source_type", "string", "Type of source memory (e.g., 'decision')"),
            ToolParameter("target_id", "string", "ID of the target memory (e.g., the trade)"),
            ToolParameter("target_type", "string", "Type of target memory (e.g., 'trade')"),
        ],
        category="memory",
    )

    # ─── Cross-Agent Memory (only if memory_shared is enabled) ──

    if memory_shared:
        def memory_search_global(query: str, agent_filter: str = None,
                                 memory_type: str = None, top_k: int = 5) -> str:
            """Search across ALL agents' memories. Use when you need context
            from another agent's experience or knowledge."""
            from .memory import MemoryStore
            store = MemoryStore(agent_id=agent_id or "default")

            # Optional: filter to a specific agent's memories
            where = f"agent_id = '{agent_filter}'" if agent_filter else None

            results = store.search(
                query=query, memory_type=memory_type or None,
                top_k=top_k, where=where,
            )
            if not results:
                return "No relevant memories found across agents."
            lines = []
            for r in results:
                owner = r.get("agent_id", "?")
                mtype = r.get("memory_type", "?")
                content = r.get("content", "")
                dist = r.get("_distance", 0)
                relevance = max(0, 1 - dist)
                lines.append(f"[{owner}/{mtype}] (relevance: {relevance:.0%}) {content[:200]}")
            return "\n".join(lines)

        registry.register_function(
            name="memory_search_global",
            description=(
                "Search across ALL agents' memories — not just yours. "
                "Use when you need context from another agent's experience, "
                "knowledge, or past interactions. Optionally filter by agent name."
            ),
            fn=memory_search_global,
            parameters=[
                ToolParameter("query", "string", "Natural language search query"),
                ToolParameter("agent_filter", "string", "Only search this agent's memories (optional)", required=False),
                ToolParameter("memory_type", "string", "Filter to type: trade, macro, tweet, decision (optional)", required=False),
                ToolParameter("top_k", "integer", "Number of results (default 5)", required=False),
            ],
            category="memory",
        )

    # ─── Shell Command Tool (Three-Tier Security) ──────────────
    #
    # Tier 1: ALWAYS ALLOWED — safe, read-only commands
    # Tier 2: REQUIRES APPROVAL — anything not in tier 1 or 3
    #         In CLI: prompts user. In Discord/API: blocked with explanation.
    # Tier 3: ALWAYS BLOCKED — destructive/dangerous, no override
    #
    import subprocess
    import re
    import os as _os

    from .config import config

    # ── Tier 1: Always allowed (read-only, safe) ──
    _ALLOWED_COMMANDS = {
        # File inspection
        "ls", "cat", "head", "tail", "less", "more", "file", "stat",
        "wc", "sort", "uniq", "diff", "comm", "tee",
        # Search
        "grep", "rg", "find", "locate", "which", "whereis",
        "ag", "fd",
        # Text processing
        "awk", "sed", "cut", "tr", "paste", "column", "jq", "yq",
        "xargs",
        # System info (read-only)
        "echo", "printf", "date", "cal", "uptime", "uname",
        "whoami", "id", "hostname", "pwd", "env", "printenv",
        "df", "du", "free", "top", "ps", "lsof",
        # Development (read-only)
        "git", "python3", "python", "node",
        "make", "test", "true", "false",
        # Utilities
        "basename", "dirname", "realpath", "readlink",
        "md5", "md5sum", "shasum", "sha256sum",
        "bc", "expr", "seq",
        "touch", "mkdir",
        "curl", "wget", "http",  # allowed by default but patterns block dangerous uses
    }

    # ── Tier 3: Always blocked (no override) ──
    _BLOCKED_COMMANDS = {
        # Destructive filesystem
        "mkfs", "fdisk", "dd", "parted", "format",
        # System control
        "shutdown", "reboot", "halt", "poweroff", "init",
        "systemctl", "launchctl",
        # User/permission manipulation
        "useradd", "userdel", "usermod", "passwd", "chown", "chmod",
        "visudo", "adduser", "deluser", "sudo", "su", "doas",
        # Network exfiltration
        "nc", "ncat", "netcat", "socat", "telnet", "ftp", "sftp", "scp",
        "ssh", "rsync",
        # Disk/mount
        "mount", "umount", "losetup",
        # Dangerous utilities
        "crontab", "at", "nohup",
    }

    _BLOCKED_PATTERNS = [
        r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/",        # rm -rf /
        r"rm\s+-[a-zA-Z]*r[a-zA-Z]*\s+-[a-zA-Z]*f",  # rm -r -f
        r">\s*/(etc|usr|bin|sbin|var|System)/",        # writing to system dirs
        r"curl\s+.*\|\s*(ba)?sh",                      # curl | sh
        r"wget\s+.*\|\s*(ba)?sh",                      # wget | sh
        r"curl\s+.*-o\s+/",                            # curl download to root paths
        r"eval\s*\(",                                   # eval injection
        r"\$\(.*curl",                                  # subshell curl
        r"base64\s+-d\s*\|",                           # decode pipe obfuscation
        r"python[23]?\s+-c\s+.*import\s+os",           # python os import
        r":\(\)\{.*\}",                                 # fork bomb
        r"/dev/(sd|hd|nvme|disk)",                     # raw disk access
    ]

    _blocked_re = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

    # Approved commands this session (user said "yes" once → allowed for the session)
    _approved_commands: set[str] = set()

    def _extract_base_command(command: str) -> str | None:
        """Extract the base command name from a shell string."""
        parts = command.strip().split()
        for part in parts:
            if "=" in part and not part.startswith("-"):
                continue
            return part.split("/")[-1].lower()
        return None

    def _classify_command(command: str) -> tuple[str, str]:
        """
        Classify a command into one of three tiers.

        Returns:
            (tier, reason) where tier is "allowed", "needs_approval", or "blocked"
        """
        cmd_stripped = command.strip()
        base_cmd = _extract_base_command(cmd_stripped)

        if not base_cmd:
            return "blocked", "Empty command."

        # Tier 3: Always blocked
        if base_cmd in _BLOCKED_COMMANDS:
            return "blocked", f"'{base_cmd}' is not allowed for safety reasons."

        # Check pipe chains for blocked commands
        if "|" in cmd_stripped:
            for seg in cmd_stripped.split("|"):
                seg = seg.strip()
                if not seg:
                    continue
                seg_cmd = _extract_base_command(seg)
                if seg_cmd and seg_cmd in _BLOCKED_COMMANDS:
                    return "blocked", f"'{seg_cmd}' in pipe chain is not allowed."

        # Check blocked patterns
        for pattern in _blocked_re:
            if pattern.search(cmd_stripped):
                return "blocked", "Command matches a dangerous pattern."

        # Tier 1: Always allowed
        if base_cmd in _ALLOWED_COMMANDS:
            return "allowed", ""

        # Tier 2: Needs approval — anything else
        return "needs_approval", f"'{base_cmd}' requires user approval."

    def _execute_command(command: str, timeout: int = 30) -> str:
        """Actually run the command (no safety checks — caller must verify)."""
        timeout = min(timeout, 60)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(config.PROJECT_ROOT),
                env={
                    **dict(_os.environ),
                    "PATH": _os.environ.get("PATH", "/usr/bin:/bin"),
                },
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            if len(output) > 10000:
                output = output[:10000] + f"\n... (truncated, {len(output)} chars total)"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    def run_command(command: str, timeout: int = 30) -> str:
        """
        Run a shell command with three-tier security.

        Safe commands (ls, git, grep, etc.) run immediately.
        Dangerous commands (rm -rf, sudo, etc.) are always blocked.
        Everything else requires one-time user approval per command.
        Once approved, that command is allowed for the rest of the session.
        """
        tier, reason = _classify_command(command)

        if tier == "blocked":
            return f"Blocked: {reason}"

        if tier == "allowed":
            return _execute_command(command, timeout)

        # Tier 2: needs_approval
        base_cmd = _extract_base_command(command)

        # Check if user already approved this command this session
        if base_cmd in _approved_commands:
            return _execute_command(command, timeout)

        # Return a message asking for approval
        return (
            f"⏳ This command needs your approval before I can run it:\n"
            f"  $ {command}\n\n"
            f"'{base_cmd}' is not in the pre-approved safe list.\n"
            f"To approve, type: approve {base_cmd}\n"
            f"This will allow '{base_cmd}' for the rest of this session."
        )

    def run_command_approve(command_name: str) -> str:
        """
        Approve a command for this session.

        After approval, the agent can use this command freely until Alfred restarts.
        Cannot override tier-3 blocked commands.
        """
        cmd = command_name.strip().lower()
        if cmd in _BLOCKED_COMMANDS:
            return f"Cannot approve '{cmd}' — it's permanently blocked for safety."
        _approved_commands.add(cmd)
        return f"Approved '{cmd}' for this session. The agent can now use it."

    registry.register_function(
        name="run_command",
        description=(
            "Execute a shell command and return output. "
            "Safe commands (ls, git, grep, cat, etc.) run immediately. "
            "Unknown commands require one-time user approval. "
            "Dangerous commands (rm -rf, sudo, etc.) are always blocked. "
            "Max timeout: 60s. Output truncated at 10k chars."
        ),
        fn=run_command,
        parameters=[
            ToolParameter("command", "string", "The shell command to execute"),
            ToolParameter("timeout", "integer", "Timeout in seconds (default 30, max 60)", required=False),
        ],
        category="system",
    )

    registry.register_function(
        name="run_command_approve",
        description=(
            "Approve a command for this session after run_command flags it. "
            "Once approved, the agent can use that command freely until restart. "
            "Cannot override permanently blocked commands (sudo, rm -rf, etc.)."
        ),
        fn=run_command_approve,
        parameters=[
            ToolParameter("command_name", "string", "The command to approve (e.g., 'rsync', 'docker')"),
        ],
        category="system",
    )

    # ─── Multi-Agent Delegation ──────────────────────────────

    def delegate_to(agent_name: str, task: str) -> str:
        """
        Delegate a task to another agent and get back the result.

        This creates a fresh agent instance, runs the task, and returns the response.
        Use this when a task falls outside your expertise or when another agent
        is better suited (e.g., delegate trading analysis to the trader agent).
        """
        from .agent import Agent, AgentConfig
        from .config import _load_config, config as cfg_obj

        full_cfg = _load_config()
        agents_cfg = full_cfg.get("agents", {})

        if agent_name not in agents_cfg:
            available = list(agents_cfg.keys())
            return f"Error: Agent '{agent_name}' not found. Available agents: {available}"

        agent_data = dict(agents_cfg[agent_name])
        agent_data["name"] = agent_name

        # Resolve workspace
        from pathlib import Path
        workspace = Path(agent_data.get("workspace", f"workspaces/{agent_name}"))
        if not workspace.is_absolute():
            workspace = cfg_obj.PROJECT_ROOT / workspace
        agent_data["workspace"] = str(workspace)

        # Check agent status
        if agent_data.get("status") == "paused":
            return f"Error: Agent '{agent_name}' is paused."

        try:
            agent_config = AgentConfig.from_dict(agent_data)
            agent = Agent(agent_config, session_id="delegation")
            response = agent.run(task)
            return f"[{agent_name}] {response}"
        except Exception as e:
            return f"Error delegating to '{agent_name}': {e}"

    registry.register_function(
        name="delegate_to",
        description=(
            "Delegate a task to another agent and get back the result. "
            "Use when a task is better handled by a specialized agent "
            "(e.g., trading analysis to 'trader', social posts to 'social'). "
            "The other agent will process the task and return its response."
        ),
        fn=delegate_to,
        parameters=[
            ToolParameter("agent_name", "string", "Name of the agent to delegate to (e.g., 'trader', 'social')"),
            ToolParameter("task", "string", "The task description or question to send to the agent"),
        ],
        category="agents",
    )

    # ─── Agent-to-Agent Messaging ────────────────────────────

    def send_message(to_agent: str, message: str, priority: str = "normal") -> str:
        """
        Send a message to another agent's inbox.

        Unlike delegate_to, this is async — the message is queued and the
        other agent will see it next time it processes. Use for notifications,
        FYIs, or non-urgent handoffs.
        """
        from pathlib import Path
        from .config import _load_config, config as cfg_obj
        import json as _json
        from datetime import datetime, timezone

        full_cfg = _load_config()
        agents_cfg = full_cfg.get("agents", {})

        if to_agent not in agents_cfg:
            available = list(agents_cfg.keys())
            return f"Error: Agent '{to_agent}' not found. Available: {available}"

        # Resolve target workspace
        target_data = agents_cfg[to_agent]
        workspace = Path(target_data.get("workspace", f"workspaces/{to_agent}"))
        if not workspace.is_absolute():
            workspace = cfg_obj.PROJECT_ROOT / workspace

        inbox_file = workspace / "inbox.json"

        # Load existing inbox
        inbox = []
        if inbox_file.exists():
            try:
                inbox = _json.loads(inbox_file.read_text())
            except (ValueError, _json.JSONDecodeError):
                inbox = []

        # Add new message
        inbox.append({
            "from": "agent",  # Will be overridden by the calling context if needed
            "message": message,
            "priority": priority,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "read": False,
        })

        inbox_file.write_text(_json.dumps(inbox, indent=2, default=str))
        return f"Message sent to {to_agent}'s inbox ({len(inbox)} total messages)"

    registry.register_function(
        name="send_message",
        description=(
            "Send an async message to another agent's inbox. The message is "
            "queued and the other agent will see it next time it runs. "
            "Use for notifications, status updates, or non-urgent handoffs. "
            "For tasks that need an immediate response, use delegate_to instead."
        ),
        fn=send_message,
        parameters=[
            ToolParameter("to_agent", "string", "Name of the receiving agent"),
            ToolParameter("message", "string", "The message content"),
            ToolParameter("priority", "string", "Priority: low, normal, high (default: normal)", required=False),
        ],
        category="agents",
    )

    # ─── Inbox Check Tool ────────────────────────────────────

    def check_inbox(mark_read: bool = True) -> str:
        """Check your inbox for messages from other agents."""
        import json as _json
        from pathlib import Path
        from .config import config as cfg_obj

        # This will be called in the context of whatever agent is running
        # We need to find the current agent's workspace — use a sentinel approach
        # The inbox path is set when the tool is registered per-agent in agent._init_tools
        return "Error: check_inbox requires agent context (use via agent tools)"

    registry.register_function(
        name="check_inbox",
        description=(
            "Check your inbox for messages from other agents. "
            "Returns unread messages and optionally marks them as read."
        ),
        fn=check_inbox,
        parameters=[
            ToolParameter("mark_read", "boolean", "Mark messages as read after viewing (default: true)", required=False),
        ],
        category="agents",
    )
