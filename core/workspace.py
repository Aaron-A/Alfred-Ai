"""
Alfred AI - Workspace Manager
Creates and manages agent workspaces with template files.
"""

from pathlib import Path
from .tools import ToolRegistry


# ─── Default Templates ───────────────────────────────────────────

SOUL_TEMPLATE = """# SOUL.md - Who You Are

You are {name}, an AI agent created by {creator}.
Your birthday is {birthday} — the day you were first brought online.

## Personality
- Direct and efficient — skip the fluff
- Opinionated when it matters
- You remember context from past interactions (via memory search)
- You admit when you don't know something
- Loyal to your creator and their mission

## Rules
- Always check your memory before answering questions about past events
- Log important decisions and outcomes to memory
- If you make a mistake, document it so you don't repeat it
- When in doubt, ask
"""

AGENTS_TEMPLATE = """# AGENTS.md - {name} Workspace

## Every Session
1. Read your system prompt (loaded automatically)
2. Check memory for recent context (automatic)
3. Execute the task at hand

## Memory
- Memories are stored in vector search and recalled automatically
- Use the `memory_store` tool to save important context for later
- Use the `memory_search` tool to look up past events

## Tools
- Your available tools are listed in TOOLS.md (auto-generated)
- Use `tool_list` to see all your tools at any time
- Use `tool_search` before creating new tools to avoid duplicates
- Use `tool_create` to build new tools when you need capabilities you don't have
- Custom tools you create go in your workspace `tools/` directory

### Creating Tools
When you create a tool with `tool_create`, follow these conventions:
- **Name**: Use snake_case (e.g. `fetch_price`, `parse_csv`)
- **Code**: Provide ONLY the function body — the boilerplate is auto-generated
- **Parameters**: Use type hints (str, int, float, bool) — they're auto-detected
- **Return type**: Always return a string (the result shown to you)
- **Dependencies**: List pip packages in the `dependencies` field (comma-separated)
- **Error handling**: Wrap external calls in try/except and return error messages
- **Always search first**: Call `tool_search` before creating to avoid duplicates

Example `code` parameter for tool_create:
```
def fetch_headline(url: str) -> str:
    \"\"\"Fetch the main headline from a web page.\"\"\"
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        h1 = soup.find("h1")
        return h1.text.strip() if h1 else "No headline found"
    except Exception as e:
        return f"Error: {{e}}"
```

## Rules
- Don't fabricate data — if you can't find it, say so
- Log important outcomes and lessons learned to memory
- Be concise in responses unless detail is requested
"""


def generate_agents_md(agent_name: str) -> str:
    """
    Generate the agent awareness section for AGENTS.md.

    Reads all agents from alfred.json and creates a "Team" section
    so this agent knows who else exists and can use delegate_to / send_message.

    The static workspace instructions (from AGENTS_TEMPLATE) are written once
    at workspace creation. This function appends a dynamic "Other Agents" block
    that is regenerated every time the agent starts.
    """
    from .config import _load_config, config as cfg_obj

    cfg = _load_config()
    agents_cfg = cfg.get("agents", {})

    # If only one agent (or none), no team section needed
    other_agents = {k: v for k, v in agents_cfg.items() if k != agent_name}
    if not other_agents:
        return ""

    lines = [
        "",
        "## Other Agents",
        "",
        "You can delegate tasks to these agents with `delegate_to` or send them",
        "async messages with `send_message`.",
        "",
    ]

    for name, acfg in other_agents.items():
        status = acfg.get("status", "active")
        desc = acfg.get("description", "")
        status_icon = "🟢" if status == "active" else "⏸️"

        line = f"- **{name}** {status_icon}"
        if desc:
            line += f" — {desc}"
        lines.append(line)

        # Add any special capabilities / notes
        if acfg.get("memory_shared"):
            lines.append(f"  - Has shared memory access")
        schedules = acfg.get("schedules", [])
        active_schedules = [s for s in schedules if s.get("enabled", True)]
        if active_schedules:
            lines.append(f"  - {len(active_schedules)} scheduled task(s)")

    lines.append("")
    return "\n".join(lines)

USER_TEMPLATE = """# USER.md - Who You're Helping

Name: {user_name}

## Preferences
- (customize this file with user-specific context)
"""


def generate_tools_md(registry: ToolRegistry, tool_names: list[str] = None) -> str:
    """
    Generate TOOLS.md content from current registry state.

    Args:
        registry: The tool registry to read from
        tool_names: Specific tool names to include (None = all)

    Returns:
        Formatted markdown string for TOOLS.md
    """
    header = (
        "# TOOLS.md - Available Tools\n\n"
        "> Auto-generated from tool registry. Do not edit manually.\n"
        "> To add custom tools, use `tool_create` or create Python files in your workspace `tools/` directory.\n"
        "> Community tools can be installed with `alfred tool install <name>` from the CLI.\n"
        "> Use `tool_list` at runtime for the most current view.\n\n"
    )

    manifest = registry.to_manifest(tool_names)
    return header + manifest + "\n"


def create_workspace(workspace_path: str, agent_name: str, overwrite: bool = False,
                     registry: ToolRegistry = None, template=None,
                     creator: str = "", birthday: str = "",
                     user_name: str = "") -> list[str]:
    """
    Create an agent workspace with template files.

    Args:
        workspace_path: Path to the workspace directory
        agent_name: Name of the agent
        overwrite: If True, overwrite existing files
        registry: Optional tool registry to generate TOOLS.md from
        template: Optional AgentTemplate with custom SOUL.md and config presets
        creator: Name of the person who created this agent
        birthday: The agent's creation date
        user_name: The user this agent serves

    Returns:
        List of created file paths
    """
    from datetime import datetime

    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "memory").mkdir(exist_ok=True)
    (workspace / "tools").mkdir(exist_ok=True)  # Workspace-local tools directory

    # Defaults
    if not birthday:
        birthday = datetime.now().strftime("%B %d, %Y")
    if not creator:
        creator = user_name or "(not set)"

    # Use template SOUL.md if provided, otherwise default
    soul_content = (template.soul_template if template else SOUL_TEMPLATE).format(
        name=agent_name, creator=creator, birthday=birthday
    )

    templates = {
        "SOUL.md": soul_content,
        "AGENTS.md": AGENTS_TEMPLATE.format(name=agent_name),
        "USER.md": USER_TEMPLATE.format(user_name=user_name or "(not set)"),
    }

    # Generate TOOLS.md from registry if available, otherwise use a placeholder
    if registry:
        templates["TOOLS.md"] = generate_tools_md(registry)
    else:
        templates["TOOLS.md"] = (
            "# TOOLS.md - Available Tools\n\n"
            "> Will be auto-generated when the agent starts.\n"
            "> Use `tool_list` at runtime to see all available tools.\n"
        )

    created = []
    for filename, content in templates.items():
        filepath = workspace / filename
        if filepath.exists() and not overwrite:
            continue
        filepath.write_text(content.strip() + "\n")
        created.append(str(filepath))

    return created
