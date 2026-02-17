"""
Alfred AI - Agent Framework
The core agent loop: perceive → remember → think → act → learn.

An Agent is:
- An LLM with a system prompt (identity, instructions, personality)
- A set of tools it can call (builtin + shared + workspace + meta)
- Access to vector memory for context recall
- A workspace with persistent files

The agent loop:
1. Receive input (user message, cron trigger, event)
2. Search memory for relevant context (automatic)
3. Build system prompt from workspace files + memories
4. Send to LLM with tools available
5. If LLM calls a tool → execute it → feed result back → loop
6. Return final response
7. Optionally store new memories from the interaction
"""

import os
import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .config import config, _load_config
from .llm import LLMClient
from .memory import MemoryStore
from .tools import ToolRegistry, get_tool_registry, register_builtin_tools


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    workspace: str  # Path to workspace directory
    description: str = ""

    # LLM settings (override global defaults)
    provider: str = ""  # Empty = use global default
    model: str = ""  # Empty = use global default

    # Which tools this agent can use
    tools: list[str] = field(default_factory=list)  # Allowlist (empty = all)
    tools_denied: list[str] = field(default_factory=list)  # Denylist (always excluded)

    # Memory settings
    memory_enabled: bool = True
    auto_memory_search: bool = True  # Automatically search memory before responding
    memory_search_top_k: int = 5

    # Agent behavior
    max_tool_rounds: int = 10  # Max tool call iterations per turn
    temperature: float = 0.7

    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Agent:
    """
    A memory-first AI agent with smart tool discovery.

    Tool loading lifecycle:
    1. Register builtins (memory_search, memory_store, run_command)
    2. Discover shared tools from tools/
    3. Discover workspace-local tools from workspaces/<name>/tools/
    4. Register meta-tools (tool_list, tool_search, tool_create, tool_remove)
    5. Auto-generate TOOLS.md from current registry state

    Usage:
        agent = Agent(AgentConfig(name="trader", workspace="./workspaces/trader"))
        response = agent.run("What TSLA setups look good today?")
    """

    def __init__(self, agent_config: AgentConfig, registry: ToolRegistry = None):
        self.config = agent_config
        self.workspace = Path(agent_config.workspace)

        # Initialize components
        self.llm = self._init_llm()
        self.memory = MemoryStore(agent_id=agent_config.name) if agent_config.memory_enabled else None

        # Each agent gets its own registry to avoid cross-contamination
        # (workspace tools are agent-specific)
        self.registry = registry or ToolRegistry()

        # ─── Tool Loading Lifecycle ─────────────────────────────
        self._init_tools()

        # Conversation history for this session
        self.history: list[dict] = []

        # Load workspace files
        self._system_prompt = None

    def _init_tools(self):
        """
        Full tool initialization lifecycle:
        1. Register builtins
        2. Discover shared tools from tools/
        3. Discover workspace-local tools
        4. Register meta-tools
        5. Auto-generate TOOLS.md
        """
        # 1. Register builtins (memory_search, memory_store, run_command)
        register_builtin_tools(self.registry)

        # 2. Discover shared tools from tools/
        from .tool_discovery import discover_shared_tools, discover_workspace_tools
        shared = discover_shared_tools(self.registry)
        if shared:
            print(f"  [tools] Loaded shared: {', '.join(shared)}")

        # 3. Discover workspace-local tools
        workspace_tools = discover_workspace_tools(self.registry, str(self.workspace))
        if workspace_tools:
            print(f"  [tools] Loaded workspace: {', '.join(workspace_tools)}")

        # 4. Register meta-tools (tool_list, tool_search, tool_create, tool_remove)
        from .tool_meta import register_meta_tools
        register_meta_tools(self.registry, self.config.name, str(self.workspace))

        # 5. Auto-generate TOOLS.md from current registry state
        self._update_tools_manifest()

    def _update_tools_manifest(self):
        """Write TOOLS.md with current tool registry state."""
        from .workspace import generate_tools_md
        available = self._get_available_tools()
        tools_md = generate_tools_md(self.registry, available)

        tools_file = self.workspace / "TOOLS.md"
        try:
            tools_file.write_text(tools_md)
        except Exception as e:
            print(f"  [tools] Warning: Could not write TOOLS.md: {e}")

    def _init_llm(self) -> LLMClient:
        """Initialize LLM client with agent-specific or global settings."""
        kwargs = {}
        if self.config.provider:
            kwargs["provider"] = self.config.provider
        if self.config.model:
            kwargs["model"] = self.config.model
        return LLMClient(**kwargs)

    @property
    def system_prompt(self) -> str:
        """Build the full system prompt from workspace files + config."""
        if self._system_prompt is None:
            self._system_prompt = self._build_system_prompt()
        return self._system_prompt

    def _build_system_prompt(self) -> str:
        """
        Build system prompt by loading workspace files in order.
        """
        parts = []

        # Core identity
        parts.append(f"You are {self.config.name}, an AI agent powered by Alfred AI.")
        if self.config.description:
            parts.append(f"Role: {self.config.description}")
        parts.append("")

        # Load workspace files in priority order
        workspace_files = [
            ("SOUL.md", "Your personality and values"),
            ("AGENTS.md", "Your workspace instructions"),
            ("USER.md", "Who you're helping"),
            ("TOOLS.md", "Available tools and how to use them"),
        ]

        for filename, label in workspace_files:
            filepath = self.workspace / filename
            if filepath.exists():
                content = filepath.read_text().strip()
                if content:
                    parts.append(f"## {label} ({filename})")
                    parts.append(content)
                    parts.append("")

        # Load today's memory file if it exists
        today = time.strftime("%Y-%m-%d")
        memory_file = self.workspace / "memory" / f"{today}.md"
        if memory_file.exists():
            content = memory_file.read_text().strip()
            if content:
                parts.append(f"## Today's Log ({today})")
                parts.append(content)
                parts.append("")

        return "\n".join(parts)

    def _get_available_tools(self) -> list[str]:
        """Get tool names available to this agent (respects allowlist + denylist)."""
        if self.config.tools:
            # Allowlist mode: only specified tools
            names = self.config.tools
        else:
            # Default: all registered tools
            names = self.registry.names()

        # Apply denylist
        if self.config.tools_denied:
            names = [n for n in names if n not in self.config.tools_denied]

        return names

    def run(self, message: str, context: dict = None) -> str:
        """
        Run the agent on a message. This is the main entry point.

        Args:
            message: User message or trigger
            context: Optional context dict (e.g., channel info, trigger source)

        Returns:
            Agent's response string
        """
        # Step 1: Auto-search memory for relevant context
        memory_context = ""
        if self.config.auto_memory_search and self.memory:
            memories = self.memory.search(
                query=message,
                top_k=self.config.memory_search_top_k,
            )
            if memories:
                memory_context = self._format_memories(memories)

        # Step 2: Build the full system prompt with memory
        full_system = self.system_prompt
        if memory_context:
            full_system += f"\n\n## Relevant Memories (auto-recalled)\n{memory_context}"

        # Step 3: Determine if we're using tools
        available_tools = self._get_available_tools()
        use_tools = len(available_tools) > 0

        # Step 4: Run the LLM loop (may involve multiple tool calls)
        if self.llm.provider == "anthropic":
            response = self._run_anthropic_loop(message, full_system, available_tools)
        else:
            response = self._run_openai_loop(message, full_system, available_tools)

        # Step 5: Add to conversation history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})

        return response

    def _run_anthropic_loop(self, message: str, system: str, tool_names: list[str]) -> str:
        """Agent loop using Anthropic's native tool_use API."""
        import anthropic

        messages = list(self.history)
        messages.append({"role": "user", "content": message})

        tools = self.registry.schemas(tool_names, format="anthropic") if tool_names else []

        for round_num in range(self.config.max_tool_rounds):
            kwargs = {
                "model": self.llm.model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            if tools:
                kwargs["tools"] = tools

            response = self.llm.anthropic_client.messages.create(**kwargs)

            # Check if the response contains tool calls
            if response.stop_reason == "tool_use":
                # Process tool calls
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        print(f"  [tool] {block.name}({json.dumps(block.input)[:80]})")
                        result = self.registry.execute(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
                continue  # Loop back for LLM to process tool results
            else:
                # Final text response
                text_parts = [b.text for b in response.content if hasattr(b, "text")]
                return "\n".join(text_parts)

        return "Error: Max tool rounds exceeded"

    def _run_openai_loop(self, message: str, system: str, tool_names: list[str]) -> str:
        """Agent loop using OpenAI-compatible API (xAI, OpenAI, Ollama)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        tools = self.registry.schemas(tool_names, format="openai") if tool_names else None

        for round_num in range(self.config.max_tool_rounds):
            # Build request
            url = f"{self.llm.base_url}/v1/chat/completions"
            payload = {
                "model": self.llm.model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": self.config.temperature,
            }
            if tools:
                payload["tools"] = tools

            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Alfred-AI/1.0",
            }
            if self.llm.api_key and self.llm.provider != "ollama":
                headers["Authorization"] = f"Bearer {self.llm.api_key}"

            import urllib.request
            import urllib.error

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")

            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                return f"LLM Error ({e.code}): {body}"
            except urllib.error.URLError as e:
                return f"Connection error: {e.reason}"

            choice = result["choices"][0]
            msg = choice["message"]

            # Check for tool calls
            if msg.get("tool_calls"):
                messages.append(msg)

                for tc in msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    print(f"  [tool] {fn_name}({json.dumps(fn_args)[:80]})")
                    tool_result = self.registry.execute(fn_name, fn_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })
                continue  # Loop back
            else:
                return msg.get("content", "")

        return "Error: Max tool rounds exceeded"

    def _format_memories(self, memories: list[dict]) -> str:
        """Format memory search results for injection into system prompt."""
        lines = []
        for i, mem in enumerate(memories, 1):
            mtype = mem.get("memory_type", "?")
            content = mem.get("content", "")
            dist = mem.get("_distance", 0)
            relevance = max(0, 1 - dist)

            line = f"{i}. [{mtype}] (relevance: {relevance:.0%}) {content[:200]}"

            # Add key structured fields based on type
            if mtype == "trade":
                symbol = mem.get("symbol", "")
                outcome = mem.get("outcome", "")
                pnl = mem.get("pnl", 0)
                if symbol:
                    line += f"\n   {symbol} {mem.get('strategy', '')} -> {outcome} (${pnl:+.2f})"
                lessons = mem.get("lessons", "")
                if lessons:
                    line += f"\n   Lessons: {lessons[:150]}"

            lines.append(line)

        return "\n".join(lines)

    def reset(self):
        """Reset conversation history (start fresh session)."""
        self.history = []
        self._system_prompt = None

    def __repr__(self):
        return (
            f"Agent(name={self.config.name!r}, "
            f"provider={self.llm.provider!r}, "
            f"model={self.llm.model!r}, "
            f"tools={len(self._get_available_tools())})"
        )


# ─── Agent Manager ──────────────────────────────────────────────

class AgentManager:
    """
    Manages multiple agents. Loads agent configs from alfred.json,
    creates agent instances on demand.
    """

    def __init__(self):
        self._agents: dict[str, Agent] = {}
        self._configs: dict[str, AgentConfig] = {}
        self._load_configs()

    def _load_configs(self):
        """Load agent configurations from alfred.json."""
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})

        for name, agent_data in agents_cfg.items():
            agent_data["name"] = name
            # Resolve workspace path
            if "workspace" in agent_data:
                workspace = Path(agent_data["workspace"])
                if not workspace.is_absolute():
                    workspace = config.PROJECT_ROOT / workspace
                agent_data["workspace"] = str(workspace)
            else:
                agent_data["workspace"] = str(config.PROJECT_ROOT / "workspaces" / name)

            self._configs[name] = AgentConfig.from_dict(agent_data)

    def get(self, name: str) -> Agent:
        """Get or create an agent by name."""
        if name not in self._agents:
            if name not in self._configs:
                raise ValueError(
                    f"Unknown agent '{name}'. "
                    f"Available: {list(self._configs.keys()) or 'none (run alfred setup)'}"
                )
            agent_config = self._configs[name]

            # Ensure workspace exists
            workspace = Path(agent_config.workspace)
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "memory").mkdir(exist_ok=True)
            (workspace / "tools").mkdir(exist_ok=True)  # Workspace tools dir

            self._agents[name] = Agent(agent_config)

        return self._agents[name]

    def create(self, agent_config: AgentConfig) -> Agent:
        """Create and register a new agent from config."""
        # Ensure workspace
        workspace = Path(agent_config.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "memory").mkdir(exist_ok=True)
        (workspace / "tools").mkdir(exist_ok=True)  # Workspace tools dir

        agent = Agent(agent_config)
        self._agents[agent_config.name] = agent
        self._configs[agent_config.name] = agent_config
        return agent

    def list(self) -> list[str]:
        """List all configured agent names."""
        return list(self._configs.keys())

    def list_details(self) -> list[dict]:
        """List all agents with details."""
        details = []
        for name, cfg in self._configs.items():
            details.append({
                "name": name,
                "description": cfg.description,
                "workspace": cfg.workspace,
                "provider": cfg.provider or config.LLM_PROVIDER,
                "model": cfg.model or config.LLM_MODEL,
            })
        return details
