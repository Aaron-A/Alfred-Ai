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
from datetime import datetime, timezone
from dataclasses import dataclass, field

from .config import config, _load_config, _save_config
from .llm import LLMClient, LLMResponse
from .memory import MemoryStore
from .tools import ToolRegistry, get_tool_registry, register_builtin_tools
from .logging import get_logger, metrics

logger = get_logger("agent")


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
    memory_shared: bool = False  # If True, agent can search across all agents' memories

    # Agent behavior
    max_tool_rounds: int = 10  # Max tool call iterations per turn
    temperature: float = 0.7

    # Session persistence
    session_max_turns: int = 50  # Max conversation turns to keep (oldest trimmed first)
    session_max_tokens: int = 80000  # Approx token budget for history (rough: 4 chars ≈ 1 token)

    # Pre-compaction flush — save expiring context to memory before trimming
    pre_compaction_flush: bool = True  # LLM summarizes expiring messages before they're dropped

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

    def __init__(self, agent_config: AgentConfig, registry: ToolRegistry = None,
                 session_id: str = None):
        """
        Initialize an agent.

        Args:
            agent_config: Agent configuration
            registry: Optional tool registry (one is created if not provided)
            session_id: Optional session scope (e.g., Discord channel ID).
                       Different session_ids get separate conversation histories.
                       Default "cli" is used for interactive chat.
        """
        self.config = agent_config
        self.workspace = Path(agent_config.workspace)

        # Initialize components
        self.llm = self._init_llm()
        self.memory = MemoryStore(agent_id=agent_config.name) if agent_config.memory_enabled else None

        # Each agent gets its own registry to avoid cross-contamination
        # (workspace tools are agent-specific)
        self.registry = registry or ToolRegistry()

        # Set workspace env var so shared tools (file_ops) know the workspace path
        os.environ["ALFRED_WORKSPACE"] = str(self.workspace)

        # ─── Tool Loading Lifecycle ─────────────────────────────
        self._init_tools()

        # Conversation history — load from disk if a previous session exists.
        # Session files are scoped: session.json (cli), session_<id>.json (discord, etc.)
        self.history: list[dict] = []
        self._session_id = session_id or "cli"
        if self._session_id == "cli":
            self._session_file = self.workspace / "session.json"
        else:
            self._session_file = self.workspace / f"session_{self._session_id}.json"
        self._session_meta: dict = {}
        self._load_session()

        # Load workspace files
        self._system_prompt = None

    def _init_tools(self):
        """
        Full tool initialization lifecycle:
        1. Register builtins (memory, run_command, delegation, messaging)
        2. Discover shared tools from tools/
        3. Discover workspace-local tools
        4. Register meta-tools
        5. Override check_inbox with workspace-aware version
        6. Auto-generate TOOLS.md
        7. Update AGENTS.md with awareness of other agents
        """
        # 1. Register builtins (memory_search, memory_store, run_command, delegate_to, send_message)
        #    Pass agent_id for memory isolation — each agent only sees its own memories.
        #    Pass memory_shared to enable cross-agent memory search (memory_search_global).
        register_builtin_tools(
            self.registry,
            agent_id=self.config.name,
            memory_shared=self.config.memory_shared,
        )

        # 2. Discover shared tools from tools/
        from .tool_discovery import discover_shared_tools, discover_workspace_tools
        shared = discover_shared_tools(self.registry)
        if shared:
            logger.debug(f"Loaded shared tools: {', '.join(shared)}")

        # 3. Discover workspace-local tools
        workspace_tools = discover_workspace_tools(self.registry, str(self.workspace))
        if workspace_tools:
            logger.debug(f"Loaded workspace tools: {', '.join(workspace_tools)}")

        # 4. Register meta-tools (tool_list, tool_search, tool_create, tool_remove)
        from .tool_meta import register_meta_tools
        register_meta_tools(self.registry, self.config.name, str(self.workspace))

        # 5. Register switch_model tool (needs self reference for runtime LLM swap)
        self._register_switch_model_tool()

        # 6. Override check_inbox with this agent's workspace path
        self._register_inbox_tool()

        # 7. Auto-generate TOOLS.md from current registry state
        self._update_tools_manifest()

        # 8. Update AGENTS.md with awareness of other agents
        self._update_agents_awareness()

    def _update_agents_awareness(self):
        """Append/update the 'Other Agents' section in AGENTS.md."""
        from .workspace import generate_agents_md

        agents_file = self.workspace / "AGENTS.md"
        if not agents_file.exists():
            return

        try:
            content = agents_file.read_text()

            # Strip any previous auto-generated section
            marker = "## Other Agents"
            if marker in content:
                content = content[:content.index(marker)].rstrip()

            # Generate fresh awareness block
            awareness = generate_agents_md(self.config.name)
            if awareness:
                content = content + "\n" + awareness

            agents_file.write_text(content)
        except Exception as e:
            logger.warning(f"Could not update AGENTS.md: {e}")

    def _register_switch_model_tool(self):
        """Register switch_model tool — lets the agent change its own LLM at runtime."""
        from .tools import ToolParameter

        agent = self  # capture for closure

        def switch_model(provider: str, model: str) -> str:
            """Switch this agent's LLM provider and model."""
            from .api import PROVIDER_MODELS

            if provider not in PROVIDER_MODELS:
                return f"Unknown provider '{provider}'. Known providers: {list(PROVIDER_MODELS.keys())}"
            if model not in PROVIDER_MODELS[provider]:
                return (
                    f"Unknown model '{model}' for {provider}. "
                    f"Available: {PROVIDER_MODELS[provider]}"
                )

            old_provider = agent.llm.provider
            old_model = agent.llm.model
            agent.switch_llm(provider, model)
            return (
                f"Switched from {old_provider}/{old_model} to {provider}/{model}. "
                f"Config saved. The new model takes effect on your next message."
            )

        self.registry.register_function(
            name="switch_model",
            description=(
                "Switch this agent's LLM provider and model. Use when the user asks "
                "to change models, switch providers, or use a different AI. "
                "The change takes effect immediately and persists across restarts."
            ),
            fn=switch_model,
            parameters=[
                ToolParameter(
                    "provider", "string",
                    "LLM provider to switch to",
                    enum=["anthropic", "xai", "openai", "ollama"],
                ),
                ToolParameter(
                    "model", "string",
                    "Model name for the chosen provider (e.g., 'claude-sonnet-4-6', 'grok-3-fast')",
                ),
            ],
            category="system",
        )

    def _register_inbox_tool(self):
        """Register workspace-aware messaging tools for this agent."""
        from .tools import ToolParameter

        inbox_file = self.workspace / "inbox.json"
        agent_name = self.config.name

        def check_inbox(mark_read: bool = True) -> str:
            """Check your inbox for messages from other agents."""
            if not inbox_file.exists():
                return "Inbox is empty."

            try:
                inbox = json.loads(inbox_file.read_text())
            except (json.JSONDecodeError, ValueError):
                return "Inbox is empty."

            if not inbox:
                return "Inbox is empty."

            unread = [m for m in inbox if not m.get("read", False)]
            if not unread:
                return f"No new messages ({len(inbox)} total, all read)."

            lines = [f"You have {len(unread)} unread message(s):"]
            for i, msg in enumerate(unread, 1):
                sender = msg.get("from", "unknown")
                priority = msg.get("priority", "normal")
                ts = msg.get("timestamp", "?")
                content = msg.get("message", "")
                pri_tag = f" [!{priority}]" if priority != "normal" else ""
                lines.append(f"\n{i}. From: {sender}{pri_tag} ({ts})")
                lines.append(f"   {content}")

            # Mark as read
            if mark_read:
                for msg in inbox:
                    msg["read"] = True
                inbox_file.write_text(json.dumps(inbox, indent=2, default=str))

            return "\n".join(lines)

        # Override the placeholder check_inbox with this workspace-aware version
        self.registry.register_function(
            name="check_inbox",
            description=(
                "Check your inbox for messages from other agents. "
                "Returns unread messages and optionally marks them as read."
            ),
            fn=check_inbox,
            parameters=[
                ToolParameter("mark_read", "boolean", "Mark messages as read (default: true)", required=False),
            ],
            category="agents",
        )

        # Override send_message to tag the sender automatically
        def send_message(to_agent: str, message: str, priority: str = "normal") -> str:
            """Send a message to another agent's inbox, tagged with your name."""
            from .config import _load_config, config as cfg_obj

            full_cfg = _load_config()
            agents_cfg = full_cfg.get("agents", {})

            if to_agent not in agents_cfg:
                available = list(agents_cfg.keys())
                return f"Error: Agent '{to_agent}' not found. Available: {available}"

            # Resolve target workspace
            target_data = agents_cfg[to_agent]
            target_ws = Path(target_data.get("workspace", f"workspaces/{to_agent}"))
            if not target_ws.is_absolute():
                target_ws = cfg_obj.PROJECT_ROOT / target_ws
            target_ws.mkdir(parents=True, exist_ok=True)

            target_inbox = target_ws / "inbox.json"

            # Load existing inbox
            inbox = []
            if target_inbox.exists():
                try:
                    inbox = json.loads(target_inbox.read_text())
                except (ValueError, json.JSONDecodeError):
                    inbox = []

            inbox.append({
                "from": agent_name,
                "message": message,
                "priority": priority,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "read": False,
            })

            target_inbox.write_text(json.dumps(inbox, indent=2, default=str))
            return f"Message sent to {to_agent}'s inbox ({len(inbox)} total messages)"

        self.registry.register_function(
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

    def _update_tools_manifest(self):
        """Write TOOLS.md with current tool registry state."""
        from .workspace import generate_tools_md
        available = self._get_available_tools()
        tools_md = generate_tools_md(self.registry, available)

        tools_file = self.workspace / "TOOLS.md"
        try:
            tools_file.write_text(tools_md)
        except Exception as e:
            logger.warning(f"Could not write TOOLS.md: {e}")

    # ─── Session Persistence ─────────────────────────────────

    def _load_session(self):
        """
        Load conversation history from disk.

        Session file format (workspace/session.json):
        {
            "agent": "alfred",
            "started_at": "2025-02-17T...",
            "last_activity": "2025-02-17T...",
            "turn_count": 42,
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }

        On load, we trim to the configured window so stale sessions
        don't blow up the context window.
        """
        if not self._session_file.exists():
            self._session_meta = {
                "agent": self.config.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "turn_count": 0,
            }
            return

        try:
            data = json.loads(self._session_file.read_text())
            messages = data.get("messages", [])

            # Load all valid messages including tool calls (critical for LLM to see
            # its own tool-calling pattern — prevents tool-call hallucination).
            # Valid roles: user, assistant (may have text or tool_use content),
            # tool (OpenAI format tool results).
            # Content can be: string, list of blocks (Anthropic), or dict (OpenAI tool msg).
            valid = [m for m in messages
                     if m.get("role") in ("user", "assistant", "tool")
                     and (m.get("content") is not None or m.get("tool_calls"))]

            # Trim to window size
            valid = self._trim_history(valid)

            self.history = valid
            # Count turns = user messages with string content (not tool_result)
            turn_count = sum(
                1 for m in valid
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            )
            self._session_meta = {
                "agent": data.get("agent", self.config.name),
                "started_at": data.get("started_at", datetime.now(timezone.utc).isoformat()),
                "last_activity": data.get("last_activity", datetime.now(timezone.utc).isoformat()),
                "turn_count": data.get("turn_count", turn_count),
            }

            if valid:
                logger.info(f"{self.config.name}: Restored {turn_count} turn(s) from previous session ({len(valid)} messages)")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"{self.config.name}: Could not load session ({e}), starting fresh")
            self._session_meta = {
                "agent": self.config.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "turn_count": 0,
            }

    def _save_session(self):
        """
        Persist current conversation history to disk.

        Called after every interaction. Writes atomically (tmp → rename)
        to avoid corruption if the process is killed mid-write.

        History now includes tool messages (assistant+tool_use, tool_result,
        tool-role messages for OpenAI). All content is pre-serialized to
        plain dicts/strings by _serialize_content() before reaching history.
        """
        self._session_meta["last_activity"] = datetime.now(timezone.utc).isoformat()
        # Count turns = number of user messages with string content (not tool_result)
        self._session_meta["turn_count"] = sum(
            1 for m in self.history
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        )

        data = {
            **self._session_meta,
            "messages": self.history,
        }

        # Atomic write: write to .tmp then rename
        tmp_file = self._session_file.with_suffix(".json.tmp")
        try:
            tmp_file.write_text(json.dumps(data, indent=2, default=str))
            tmp_file.rename(self._session_file)
        except Exception as e:
            logger.warning(f"{self.config.name}: Could not save session: {e}")
            # Clean up temp file on failure
            tmp_file.unlink(missing_ok=True)

    def _split_into_turns(self, messages: list[dict]) -> list[list[dict]]:
        """
        Split a flat message list into logical turns.

        A "turn" starts with a user message (with string content — not a tool_result)
        and includes everything up to (but not including) the next such user message.
        This groups: user → assistant+tool_use → tool_result → ... → final assistant.

        Returns a list of turns, where each turn is a list of messages.
        """
        turns = []
        current_turn = []

        for msg in messages:
            # Detect the start of a new human turn (not a tool_result user message)
            is_new_human_turn = (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
            )

            if is_new_human_turn and current_turn:
                turns.append(current_turn)
                current_turn = []

            current_turn.append(msg)

        if current_turn:
            turns.append(current_turn)

        return turns

    def _trim_history(self, messages: list[dict]) -> list[dict]:
        """
        Intelligently trim conversation history to stay within limits.

        Strategy:
        1. Split messages into logical turns (user msg + all tool calls + final response)
        2. Hard cap on turn count (session_max_turns)
        3. Token budget — estimate token usage, trim oldest turns first
        4. PRE-COMPACTION FLUSH: Before dropping messages, summarize and store
           them in vector memory so important context isn't lost.

        The flush prevents context amnesia during long sessions. When the trader
        runs 30+ tool calls analyzing BTC, the early analysis gets preserved
        in memory before it falls off the context window.
        """
        max_turns = self.config.session_max_turns
        max_chars = self.config.session_max_tokens * 4  # Rough: 1 token ≈ 4 chars

        turns = self._split_into_turns(messages)
        turns_to_drop = []

        # 1. Trim by turn count
        if len(turns) > max_turns:
            turns_to_drop.extend(turns[:-max_turns])
            turns = turns[-max_turns:]

        # 2. Trim by estimated token budget
        def turn_chars(turn):
            return sum(self._estimate_content_chars(m.get("content", "")) for m in turn)

        total_chars = sum(turn_chars(t) for t in turns)
        while total_chars > max_chars and len(turns) > 1:
            dropped = turns.pop(0)
            turns_to_drop.append(dropped)
            total_chars -= turn_chars(dropped)

        # 3. Pre-compaction flush — save expiring context to memory
        if turns_to_drop and self.memory and self.config.pre_compaction_flush:
            # Flatten dropped turns back into a message list
            dropped_messages = [m for turn in turns_to_drop for m in turn]
            self._flush_expiring_context(dropped_messages)

        # Flatten remaining turns back into a flat message list
        return [m for turn in turns for m in turn]

    # ─── Pre-Compaction Flush ─────────────────────────────────

    # Compact system prompt for the flush summarizer — kept minimal to save tokens
    _FLUSH_SYSTEM_PROMPT = (
        "You are a context preservation assistant. You will receive conversation messages "
        "that are about to be removed from an AI agent's context window. "
        "Extract and summarize ONLY the important information that should be remembered:\n"
        "- Decisions made and their reasoning\n"
        "- Trade entries, exits, stops, targets, and P&L\n"
        "- Key analysis results (indicator values, support/resistance levels)\n"
        "- Position updates and risk management actions\n"
        "- Errors encountered and lessons learned\n"
        "- Any commitments or plans stated\n\n"
        "Be concise. Use bullet points. Skip pleasantries, tool call noise, and "
        "repetitive data fetches. Only preserve what the agent would need to recall later.\n"
        "If there is nothing important to preserve, respond with exactly: NOTHING_TO_FLUSH"
    )

    def _flush_expiring_context(self, expiring_messages: list[dict]):
        """
        Summarize expiring messages and store the summary in vector memory.

        Makes a lightweight LLM call with just the expiring messages and a fixed
        summarization prompt. No tools, no memory search — just extract and store.

        This runs synchronously during trim, adding ~1-3 seconds per flush event.
        Flushes happen rarely (every ~50 turns or when token budget is exceeded).
        """
        try:
            # Build a compact text representation of expiring messages
            lines = []
            for msg in expiring_messages:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if not content:
                    continue

                # Convert structured content (tool_use blocks, tool_results) to text
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                parts.append(f"[tool_call: {block.get('name', '?')}({json.dumps(block.get('input', {}))[:200]})]")
                            elif block.get("type") == "tool_result":
                                result_text = str(block.get("content", ""))[:500]
                                parts.append(f"[tool_result: {result_text}]")
                    content = " ".join(parts)
                elif not isinstance(content, str):
                    content = str(content)

                # Truncate very long messages (tool results can be huge)
                if len(content) > 1000:
                    content = content[:1000] + "... [truncated]"
                lines.append(f"[{role}]: {content}")

            if not lines:
                return

            expiring_text = "\n".join(lines)

            # Cap the total flush input to avoid expensive calls on massive sessions
            if len(expiring_text) > 8000:
                expiring_text = expiring_text[:8000] + "\n... [truncated — too many messages]"

            # Make a lightweight LLM call — no tools, just summarize
            if self.llm.provider == "anthropic":
                summary = self._flush_anthropic(expiring_text)
            else:
                summary = self._flush_openai(expiring_text)

            # If the LLM says nothing worth keeping, skip storage
            if not summary or "NOTHING_TO_FLUSH" in summary:
                logger.debug(f"{self.config.name}: flush found nothing worth preserving")
                return

            # Store the summary in memory
            from models.base import MemoryRecord
            record = MemoryRecord(
                content=summary,
                memory_type="context_flush",
                tags="auto,pre-compaction,session-context",
                agent_id=self.config.name,
                source="pre_compaction_flush",
            )
            self.memory.store(record, dedup=False)  # Never dedup flushes — each is unique

            msg_count = len(expiring_messages)
            logger.info(f"{self.config.name}: flushed {msg_count} expiring messages to memory")

        except Exception as e:
            # Flush is best-effort — never crash the agent over it
            logger.warning(f"{self.config.name}: pre-compaction flush failed (non-fatal): {e}")

    def _flush_anthropic(self, expiring_text: str) -> str:
        """Flush via Anthropic API — minimal call, no tools."""
        try:
            response = self.llm.anthropic_client.messages.create(
                model=self.llm.model,
                max_tokens=1024,
                temperature=0.3,  # Low temp for factual summarization
                system=self._FLUSH_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Summarize the important context from these expiring messages:\n\n{expiring_text}",
                }],
            )
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_parts).strip()
        except Exception as e:
            logger.warning(f"Flush LLM call failed: {e}")
            return ""

    def _flush_openai(self, expiring_text: str) -> str:
        """Flush via OpenAI-compatible API — minimal call, no tools."""
        import urllib.request
        import urllib.error

        url = f"{self.llm.base_url}/v1/chat/completions"
        payload = {
            "model": self.llm.model,
            "max_tokens": 1024,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": self._FLUSH_SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize the important context from these expiring messages:\n\n{expiring_text}"},
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Alfred-AI/1.0",
        }
        if self.llm.api_key and self.llm.provider != "ollama":
            headers["Authorization"] = f"Bearer {self.llm.api_key}"

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"].get("content", "").strip()
        except Exception as e:
            logger.warning(f"Flush LLM call failed: {e}")
            return ""

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
        """Build the full system prompt from workspace files + config.

        Re-reads workspace files every time so edits to SOUL.md, AGENTS.md,
        etc. take effect without restarting the daemon.
        """
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
            ("AGENTS.md", "Your workspace instructions and team"),
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

        # Check inbox for unread messages
        inbox_file = self.workspace / "inbox.json"
        if inbox_file.exists():
            try:
                inbox = json.loads(inbox_file.read_text())
                unread = [m for m in inbox if not m.get("read", False)]
                if unread:
                    parts.append(f"## Inbox ({len(unread)} unread)")
                    parts.append("You have unread messages. Use `check_inbox` to read them.")
                    parts.append("")
            except (json.JSONDecodeError, ValueError):
                pass

        return "\n".join(parts)

    # ─── Message Serialization Helpers ─────────────────────

    @staticmethod
    def _serialize_content(content):
        """
        Serialize Anthropic SDK content blocks to plain dicts for JSON storage.

        Handles:
        - str content → pass through
        - list of SDK objects (TextBlock, ToolUseBlock) → list of dicts
        - list of dicts (already serialized or tool_result) → pass through
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            result = []
            for block in content:
                if isinstance(block, dict):
                    result.append(block)
                elif hasattr(block, "model_dump"):
                    # Anthropic SDK object (TextBlock, ToolUseBlock, etc.)
                    result.append(block.model_dump())
                else:
                    result.append(block)
            return result
        return content

    @staticmethod
    def _estimate_content_chars(content) -> int:
        """Estimate character count for content (string or list of blocks)."""
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    # tool_result, text block, or tool_use block
                    total += len(str(block.get("content", "")))
                    total += len(str(block.get("text", "")))
                    total += len(str(block.get("input", "")))
                elif isinstance(block, str):
                    total += len(block)
                elif hasattr(block, "text"):
                    total += len(getattr(block, "text", ""))
            return total
        return 0

    # Tools the LLM must never see (user-only tools)
    _INTERNAL_TOOLS = {"run_command_approve"}

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

        # Always hide internal/user-only tools from the LLM
        names = [n for n in names if n not in self._INTERNAL_TOOLS]

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
        start_time = time.monotonic()

        # Apply any pending LLM switch from a prior turn's switch_model tool call
        self._apply_pending_llm_switch()

        try:
            # Step 1: Auto-search memory for relevant context (scoped to this agent)
            memory_context = ""
            if self.config.auto_memory_search and self.memory:
                memories = self.memory.search(
                    query=message,
                    top_k=self.config.memory_search_top_k,
                    where=f"agent_id = '{self.config.name}'",
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
            # Returns (response_text, tool_call_count, input_tokens, output_tokens, new_messages)
            # new_messages contains the FULL chain: user → [assistant+tool_use → tool_result]* → final assistant
            if self.llm.provider == "anthropic":
                response, tool_call_count, input_tokens, output_tokens, new_messages = \
                    self._run_anthropic_loop(message, full_system, available_tools)
            else:
                response, tool_call_count, input_tokens, output_tokens, new_messages = \
                    self._run_openai_loop(message, full_system, available_tools)

            # Step 5: Add FULL message chain to history (preserves tool calls)
            # This is critical: the LLM needs to see its own tool calls in prior turns
            # to maintain the "I must call tools to perform actions" behavior pattern.
            self.history.extend(new_messages)

            # Trim if we've grown past the window, then save to disk
            self.history = self._trim_history(self.history)
            self._save_session()

            # Step 6: Record metrics (with token usage)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            metrics.record_message(
                self.config.name,
                elapsed_ms=elapsed_ms,
                tool_calls=tool_call_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider=self.llm.provider,
                model=self.llm.model,
            )
            token_info = f", {input_tokens}+{output_tokens} tokens" if input_tokens else ""
            logger.info(f"{self.config.name}: responded in {elapsed_ms}ms{token_info}")

            return response

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            metrics.record_error(self.config.name, str(e),
                                 provider=self.llm.provider, model=self.llm.model)
            logger.error(f"{self.config.name}: error after {elapsed_ms}ms: {e}")
            raise

    def _run_anthropic_loop(self, message: str, system: str, tool_names: list[str]) -> tuple[str, int, int, int, list[dict]]:
        """
        Agent loop using Anthropic's native tool_use API.

        Returns:
            (response_text, tool_call_count, total_input_tokens, total_output_tokens, new_messages)
            new_messages: all messages added this turn (user + tool calls + tool results + final assistant)
                          — serialized to plain dicts for JSON persistence.
        """
        import anthropic

        messages = list(self.history)
        messages.append({"role": "user", "content": message})

        # Track which messages are NEW this turn (for persisting to history)
        history_start_idx = len(self.history)

        tools = self.registry.schemas(tool_names, format="anthropic") if tool_names else []

        total_input_tokens = 0
        total_output_tokens = 0
        tool_call_count = 0

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

            # Accumulate token usage across rounds
            usage = getattr(response, "usage", None)
            if usage:
                total_input_tokens += getattr(usage, "input_tokens", 0)
                total_output_tokens += getattr(usage, "output_tokens", 0)

            # Check if the response contains tool calls
            if response.stop_reason == "tool_use":
                # Process tool calls
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_call_count += 1
                        logger.info(f"{self.config.name}: tool {block.name}({json.dumps(block.input)[:200]})")
                        result = self.registry.execute(block.name, block.input)
                        # Log errors/failures so they're visible in logs
                        if result and (result.startswith("Error") or "HTTP 4" in result[:20] or "HTTP 5" in result[:20]):
                            logger.warning(f"{self.config.name}: tool {block.name} returned: {result[:200]}")
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
                response_text = "\n".join(text_parts)

                # Append the final assistant text message
                messages.append({"role": "assistant", "content": response_text})

                # Extract and serialize all NEW messages from this turn
                new_messages = []
                for msg in messages[history_start_idx:]:
                    new_messages.append({
                        "role": msg["role"],
                        "content": self._serialize_content(msg["content"]),
                    })

                return response_text, tool_call_count, total_input_tokens, total_output_tokens, new_messages

        # Max rounds exceeded — still return what we have
        new_messages = []
        for msg in messages[history_start_idx:]:
            new_messages.append({
                "role": msg["role"],
                "content": self._serialize_content(msg["content"]),
            })
        return "Error: Max tool rounds exceeded", tool_call_count, total_input_tokens, total_output_tokens, new_messages

    def _run_openai_loop(self, message: str, system: str, tool_names: list[str]) -> tuple[str, int, int, int, list[dict]]:
        """
        Agent loop using OpenAI-compatible API (xAI, OpenAI, Ollama).

        Returns:
            (response_text, tool_call_count, total_input_tokens, total_output_tokens, new_messages)
            new_messages: all messages added this turn (user + tool calls + tool results + final assistant)
                          — plain dicts ready for JSON persistence.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        # OpenAI format: history may contain "tool" role messages from prior turns.
        # Filter those out when building the API messages — OpenAI needs the
        # full assistant+tool_calls message before any tool-role messages.
        messages.extend(self._openai_history_messages())
        messages.append({"role": "user", "content": message})

        # Track where new messages start (after system + history + user)
        # For history purposes, we only want the user message onward
        new_messages = [{"role": "user", "content": message}]

        tools = self.registry.schemas(tool_names, format="openai") if tool_names else None

        total_input_tokens = 0
        total_output_tokens = 0
        tool_call_count = 0

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
                return f"LLM Error ({e.code}): {body}", tool_call_count, total_input_tokens, total_output_tokens, new_messages
            except urllib.error.URLError as e:
                return f"Connection error: {e.reason}", tool_call_count, total_input_tokens, total_output_tokens, new_messages

            # Accumulate token usage
            usage = result.get("usage", {})
            total_input_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)

            choice = result["choices"][0]
            msg = choice["message"]

            # Check for tool calls
            if msg.get("tool_calls"):
                messages.append(msg)
                new_messages.append(msg)  # Preserve assistant+tool_calls message

                for tc in msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    tool_call_count += 1
                    logger.info(f"{self.config.name}: tool {fn_name}({json.dumps(fn_args)[:200]})")
                    tool_result = self.registry.execute(fn_name, fn_args)
                    # Log errors/failures so they're visible in logs
                    if tool_result and (tool_result.startswith("Error") or "HTTP 4" in tool_result[:20] or "HTTP 5" in tool_result[:20]):
                        logger.warning(f"{self.config.name}: tool {fn_name} returned: {tool_result[:200]}")

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }
                    messages.append(tool_msg)
                    new_messages.append(tool_msg)
                continue  # Loop back
            else:
                response_text = msg.get("content", "")
                final_msg = {"role": "assistant", "content": response_text}
                new_messages.append(final_msg)
                return response_text, tool_call_count, total_input_tokens, total_output_tokens, new_messages

        return "Error: Max tool rounds exceeded", tool_call_count, total_input_tokens, total_output_tokens, new_messages

    def _openai_history_messages(self) -> list[dict]:
        """
        Build OpenAI-format history messages from self.history.

        OpenAI format requires that tool-role messages follow their
        corresponding assistant message with tool_calls. Our history
        already stores them in the correct order.
        """
        msgs = []
        for msg in self.history:
            role = msg.get("role")
            if role in ("user", "assistant", "tool"):
                msgs.append(msg)
        return msgs

    def run_stream(self, message: str, context: dict = None):
        """
        Stream agent's response token by token.

        NOTE: Streaming bypasses the tool loop — the LLM generates a direct
        text response without tool calls. Use run() for tool-enabled interactions.

        Args:
            message: User message
            context: Optional context dict

        Yields:
            str: Text chunks as they arrive from the LLM
        """
        start_time = time.monotonic()

        try:
            # Step 1: Memory search
            memory_context = ""
            if self.config.auto_memory_search and self.memory:
                memories = self.memory.search(
                    query=message,
                    top_k=self.config.memory_search_top_k,
                    where=f"agent_id = '{self.config.name}'",
                )
                if memories:
                    memory_context = self._format_memories(memories)

            # Step 2: Build system prompt
            full_system = self.system_prompt
            if memory_context:
                full_system += f"\n\n## Relevant Memories (auto-recalled)\n{memory_context}"

            # Step 3: Stream from LLM (no tools — direct response)
            full_response = []
            for chunk in self.llm.stream(
                prompt=message,
                system=full_system,
                context=self.history,
                max_tokens=4096,
                temperature=self.config.temperature,
            ):
                full_response.append(chunk)
                yield chunk

            # Step 4: Save to history
            response_text = "".join(full_response)
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": response_text})
            self.history = self._trim_history(self.history)
            self._save_session()

            # Step 5: Metrics
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            metrics.record_message(self.config.name, elapsed_ms=elapsed_ms,
                                   provider=self.llm.provider, model=self.llm.model)
            logger.info(f"{self.config.name}: streamed response in {elapsed_ms}ms")

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            metrics.record_error(self.config.name, str(e))
            logger.error(f"{self.config.name}: stream error after {elapsed_ms}ms: {e}")
            raise

    def _format_memories(self, memories: list[dict]) -> str:
        """Format memory search results for injection into system prompt."""
        lines = [
            "NOTE: These are summaries from past runs — they are NOT evidence of tool calls.",
            "You must still call the actual tool (x_post_tweet, http_request, etc.) to perform actions.",
            ""
        ]
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
        """Reset conversation history and clear saved session."""
        self.history = []
        self._system_prompt = None
        self._session_meta = {
            "agent": self.config.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "turn_count": 0,
        }
        # Remove saved session file
        self._session_file.unlink(missing_ok=True)

    def switch_llm(self, provider: str, model: str):
        """Switch LLM provider/model and persist to alfred.json.

        The config change is saved immediately and takes effect on the **next**
        run() call.  We defer the in-memory client swap because this tool
        may be called mid-loop — changing self.llm while the current loop
        is talking to the old provider would corrupt the conversation.
        """
        # 1. Stage the pending swap (applied in _apply_pending_llm_switch)
        self._pending_llm_switch = (provider, model)

        # 2. Update agent config (in-memory) — informational only
        self.config.provider = provider
        self.config.model = model

        # 3. Persist to alfred.json so it survives restarts
        cfg = _load_config()
        cfg.setdefault("agents", {}).setdefault(self.config.name, {}).update({
            "provider": provider,
            "model": model,
        })
        _save_config(cfg)

        logger.info(f"{self.config.name}: queued LLM switch to {provider}/{model}")

    def _apply_pending_llm_switch(self):
        """Apply any pending LLM switch. Called at the start of run()."""
        pending = getattr(self, "_pending_llm_switch", None)
        if pending is None:
            return
        provider, model = pending
        self._pending_llm_switch = None

        self.llm.provider = provider
        self.llm.model = model
        self.llm.api_key = config.get_api_key(provider)
        self.llm.base_url = LLMClient.PROVIDER_URLS.get(provider, "")
        self.llm._anthropic_client = None  # force re-init
        logger.info(f"{self.config.name}: LLM switch applied — now {provider}/{model}")

    @property
    def session_info(self) -> dict:
        """Get session metadata for status display."""
        return {
            **self._session_meta,
            "history_length": len(self.history),
            "turns": sum(1 for m in self.history if m.get("role") == "user" and isinstance(m.get("content"), str)),
        }

    # ─── Session Management ──────────────────────────────────

    @staticmethod
    def list_sessions(workspace_path: str) -> list[dict]:
        """
        List all saved sessions for an agent workspace.

        Returns a list of session info dicts sorted by last activity (newest first):
        [
            {"session_id": "cli", "file": "session.json", "turns": 12, ...},
            {"session_id": "1473423154759074026", "file": "session_147...json", ...},
        ]
        """
        workspace = Path(workspace_path)
        sessions = []

        if not workspace.exists():
            return sessions

        # Match session.json and session_*.json
        session_files = list(workspace.glob("session.json")) + list(workspace.glob("session_*.json"))
        for session_file in sorted(set(session_files), key=lambda f: f.name, reverse=True):
            # Skip temp files from atomic writes
            if ".tmp" in session_file.name:
                continue

            try:
                data = json.loads(session_file.read_text())
                messages = data.get("messages", [])

                # Derive session_id from filename
                name = session_file.stem  # "session" or "session_abc123"
                if name == "session":
                    session_id = "cli"
                else:
                    session_id = name.replace("session_", "", 1)

                sessions.append({
                    "session_id": session_id,
                    "file": session_file.name,
                    "agent": data.get("agent", "?"),
                    "started_at": data.get("started_at", ""),
                    "last_activity": data.get("last_activity", ""),
                    "turn_count": data.get("turn_count", len(messages) // 2),
                    "message_count": len(messages),
                })
            except (json.JSONDecodeError, KeyError, OSError):
                # Corrupted session file — include it but mark as broken
                sessions.append({
                    "session_id": session_file.stem.replace("session_", "", 1) if "_" in session_file.stem else "cli",
                    "file": session_file.name,
                    "agent": "?",
                    "started_at": "",
                    "last_activity": "",
                    "turn_count": 0,
                    "message_count": 0,
                    "error": "corrupt",
                })

        # Sort by last_activity descending (most recent first)
        sessions.sort(key=lambda s: s.get("last_activity", ""), reverse=True)
        return sessions

    @staticmethod
    def get_session_messages(workspace_path: str, session_id: str) -> list[dict]:
        """
        Load raw messages from a saved session file.

        Returns list of {"role": "user"|"assistant", "content": "..."} dicts.
        """
        workspace = Path(workspace_path)
        if session_id == "cli":
            session_file = workspace / "session.json"
        else:
            session_file = workspace / f"session_{session_id}.json"

        if not session_file.exists():
            return []

        try:
            data = json.loads(session_file.read_text())
            messages = data.get("messages", [])
            return [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def export_session(workspace_path: str, session_id: str, format: str = "markdown") -> str:
        """
        Export a session as formatted text.

        Args:
            workspace_path: Path to agent workspace
            session_id: Session to export ("cli", channel ID, etc.)
            format: "markdown" or "text"

        Returns:
            Formatted string of the conversation.
        """
        workspace = Path(workspace_path)
        if session_id == "cli":
            session_file = workspace / "session.json"
        else:
            session_file = workspace / f"session_{session_id}.json"

        if not session_file.exists():
            return ""

        try:
            data = json.loads(session_file.read_text())
        except (json.JSONDecodeError, OSError):
            return ""

        messages = data.get("messages", [])
        agent_name = data.get("agent", "assistant")
        started = data.get("started_at", "unknown")
        turns = data.get("turn_count", len(messages) // 2)

        if format == "markdown":
            lines = [
                f"# Conversation with {agent_name}",
                f"",
                f"- **Session:** {session_id}",
                f"- **Started:** {started}",
                f"- **Turns:** {turns}",
                f"",
                f"---",
                f"",
            ]
            for msg in messages:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"### You")
                    lines.append(f"{content}")
                    lines.append("")
                elif role == "assistant":
                    lines.append(f"### {agent_name.title()}")
                    lines.append(f"{content}")
                    lines.append("")
            return "\n".join(lines)

        else:  # plain text
            lines = [
                f"Conversation with {agent_name} | Session: {session_id} | Started: {started} | Turns: {turns}",
                "=" * 60,
                "",
            ]
            for msg in messages:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"You: {content}")
                    lines.append("")
                elif role == "assistant":
                    lines.append(f"{agent_name.title()}: {content}")
                    lines.append("")
            return "\n".join(lines)

    @staticmethod
    def delete_session(workspace_path: str, session_id: str) -> bool:
        """
        Delete a saved session file.

        Returns True if deleted, False if not found.
        """
        workspace = Path(workspace_path)
        if session_id == "cli":
            session_file = workspace / "session.json"
        else:
            session_file = workspace / f"session_{session_id}.json"

        if session_file.exists():
            session_file.unlink()
            return True
        return False

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
