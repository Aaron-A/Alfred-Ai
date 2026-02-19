"""
Alfred AI - CLI Entry Point
Run with: alfred <command>

Commands:
    setup              - Interactive setup wizard
    start              - Start Alfred (Discord bot + services)
    stop               - Stop Alfred
    status             - Show current configuration and running state
    logs               - Tail the Alfred log file

    provider add       - Add an LLM provider
    models update      - Fetch latest models from provider APIs
    models list        - Show available models

    tools list         - List all available tools (builtin + shared)
    tools list <agent> - List tools available to a specific agent

    agent create       - Create a new agent
    agent list         - List all agents
    agent info         - Show agent details
    agent chat         - Interactive chat with an agent
    agent pause        - Pause an agent
    agent resume       - Resume a paused agent
    agent delete       - Delete an agent

    agent schedule add    - Add a scheduled task
    agent schedule list   - List scheduled tasks
    agent schedule remove - Remove a scheduled task

    service add        - Add credentials for an external service (e.g. alpaca)
    service list       - Show configured services with masked keys
    service remove     - Remove a service configuration

    discord setup      - Configure Discord bot (token, guild, channel→agent mapping)
    discord status     - Show current Discord configuration
    discord channel add    - Map a new Discord channel to an agent
    discord channel remove - Remove a channel mapping
    discord channel list   - Show current channel→agent mappings

    api start          - Start the HTTP API server (default port 7700)

    demo               - Run the memory demo
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))


def print_usage():
    print("""
Alfred AI - Memory-First Agent Framework

Usage: alfred <command>

Commands:
    setup                               Interactive setup wizard
    start                               Start all services (API + scheduler + Discord)
    start --fg                          Start in foreground (Ctrl+C to stop)
    start --port 8080                   Start with API on custom port
    stop                                Stop all services
    status                              Show configuration and running state
    logs                                Tail the log file

    provider add <name>                 Add an LLM provider (anthropic, xai, openai, ollama)

    models update [provider]            Fetch latest models from provider APIs
    models list [provider]              Show available models (cached)

    agent create <name>                 Create a new agent with workspace
    agent list                          List all configured agents
    agent info <name>                   Show detailed agent info
    agent chat <name> [--session NAME]   Interactive chat with an agent
    agent pause <name>                  Pause an agent (disables schedules)
    agent resume <name>                 Resume a paused agent
    agent delete <name>                 Delete an agent and its config

    agent schedule add <name>           Add a scheduled task to an agent
    agent schedule list [name]          List scheduled tasks (all or one agent)
    agent schedule remove <name> <id>   Remove a scheduled task
    agent schedule enable <name> <id>   Resume a paused schedule
    agent schedule disable <name> <id>  Pause a schedule without removing it
    agent schedule run <name> <id>      Manually trigger a schedule right now
    agent schedule history <name> <id>  Show run history with stats
    agent schedule retry <name> <id>    Configure retry settings

    session list [agent]                List all saved conversation sessions
    session view <agent> <id>           View conversation history
    session export <agent> <id>         Export a session to markdown
    session delete <agent> <id>         Delete a saved session

    tools list [agent_name]             List all registered tools
                                        (omit name for global view, add name for agent-specific)

    service add <name>                  Add credentials for an external service
    service list                        Show configured services (masked keys)
    service remove <name>               Remove a service configuration

    discord setup                       Configure Discord bot (token, channels, agents)
    discord status                      Show Discord configuration
    discord channel add                 Map a new channel to an agent
    discord channel remove              Remove a channel mapping
    discord channel list                Show channel→agent mappings

    api start [--port PORT]             Start API only — dev mode (default: 7700)

    demo                                Run the memory layer demo

Examples:
    alfred setup                                    # First-time setup
    alfred start                                    # Start all services (API, scheduler, Discord)
    alfred start --fg                               # Start in foreground (Ctrl+C to stop)
    alfred stop                                     # Stop all services
    alfred status                                   # Check what's running
    alfred logs                                     # Watch the log
    alfred provider add anthropic                   # Add Claude as a provider
    alfred provider add brave                       # Add Brave Search (free web search)
    alfred agent create trader                      # Create a trading agent
    alfred agent chat trader                        # Chat with it
    alfred agent chat trader --session research     # Named session
    alfred session list                             # See all saved sessions
    alfred session view alfred cli                  # View CLI conversation history
    alfred discord setup                            # Configure Discord channels
    alfred api start                                # Start REST API on port 7700
    alfred api start --port 8080                    # Start on custom port
""")


def cmd_setup():
    from cli.setup import run_setup
    run_setup()


def cmd_status():
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config
    from core.llm import detect_ollama

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Alfred is not configured yet.[/]")
        console.print("  Run: [bold]alfred setup[/]\n")
        return

    cfg = _load_config()
    console.print()

    table = Table(
        box=box.ROUNDED,
        title="Alfred AI Status",
        title_style="bold cyan",
    )
    table.add_column("Component", style="bold")
    table.add_column("Value")
    table.add_column("Status")

    # LLM - Primary / Secondary
    llm = cfg.get("llm", {})
    primary = llm.get("primary", {})
    secondary = llm.get("secondary", {})

    # Handle old flat format
    if not primary and "provider" in llm:
        primary = {"provider": llm["provider"], "model": llm.get("model", "?")}

    if primary:
        table.add_row(
            "Primary LLM",
            f"{primary.get('provider', '?')} / {primary.get('model', '?')}",
            "[green]active[/]",
        )
    else:
        table.add_row("Primary LLM", "not set", "[yellow]run: alfred provider add <name>[/]")

    if secondary:
        table.add_row(
            "Secondary LLM",
            f"{secondary.get('provider', '?')} / {secondary.get('model', '?')}",
            "[green]fallback[/]",
        )

    # Providers
    for pid, pcfg in cfg.get("providers", {}).items():
        has_key = "api_key" in pcfg and pcfg["api_key"]
        if pid == "ollama":
            running, models = detect_ollama()
            status = f"[green]running ({len(models)} models)[/]" if running else "[red]not running[/]"
        elif has_key:
            status = "[green]key set[/]"
        else:
            status = "[red]no key[/]"

        # Show role
        role = ""
        if pid == primary.get("provider"):
            role = " [cyan](primary)[/]"
        elif pid == secondary.get("provider"):
            role = " [dim](secondary)[/]"

        table.add_row(f"  {pid}", f"{pcfg.get('model', '?')}{role}", status)

    # Embeddings
    emb = cfg.get("embeddings", {})
    table.add_row("Embeddings", emb.get("model", "?"), "[green]local[/]")

    # Memory
    table.add_row("Vector Store", "LanceDB", "[green]data/lancedb/[/]")

    # API server
    import socket
    api_port = 7700
    try:
        with socket.create_connection(("localhost", api_port), timeout=1):
            table.add_row("API Server", f"port {api_port}", f"[bold green]listening[/]  →  http://localhost:{api_port}")
    except (ConnectionRefusedError, OSError):
        table.add_row("API Server", f"port {api_port}", "[dim]not running[/]")

    # Discord
    discord_cfg = cfg.get("discord", {})
    if discord_cfg.get("bot_token"):
        from core.discord import is_bot_running, is_bot_healthy
        pid = is_bot_running()
        channels = discord_cfg.get("channels", {})
        channel_list = ", ".join(
            f"#{c.get('name', '?')}→{c.get('agent', '?')}"
            for c in channels.values()
        )
        if pid:
            healthy, health_msg = is_bot_healthy()
            if healthy:
                status_str = f"[bold green]{health_msg}[/]"
            else:
                status_str = f"[bold red]{health_msg}[/]"
        elif channels:
            status_str = f"[dim]stopped[/] — {channel_list}"
        else:
            status_str = "[yellow]no channels[/]"
        table.add_row(
            "Discord Bot",
            f"{len(channels)} channel(s)",
            status_str,
        )
    else:
        table.add_row("Discord Bot", "not configured", "[dim]run: alfred discord setup[/]")

    # Agents
    agents = cfg.get("agents", {})
    if agents:
        for name, acfg in agents.items():
            agent_status = acfg.get("status", "active")
            schedules = acfg.get("schedules", [])
            sched_count = len([s for s in schedules if s.get("enabled", True)])

            agent_provider = acfg.get("provider", "")
            if agent_provider:
                agent_model = acfg.get("model", cfg.get("providers", {}).get(agent_provider, {}).get("model", "?"))
                llm_info = f"{agent_provider} / {agent_model}"
            else:
                llm_info = "(uses primary)"

            if agent_status == "paused":
                status_str = "[yellow]paused[/]"
            else:
                status_str = "[green]active[/]"
            if sched_count > 0:
                status_str += f" [dim]({sched_count} schedule{'s' if sched_count != 1 else ''})[/]"

            table.add_row(f"  Agent: {name}", llm_info, status_str)
    else:
        table.add_row("Agents", "none", "[dim]run: alfred agent create <name>[/]")

    console.print(table)
    console.print()


def cmd_agent_create(name: str):
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.rule import Rule
    from core.config import CONFIG_FILE, _load_config, _save_config, config
    from core.workspace import create_workspace

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()

    # Check if agent already exists
    if name in cfg.get("agents", {}):
        console.print(f"\n  [yellow]Agent '{name}' already exists.[/]")
        if not Confirm.ask("  Overwrite?", default=False):
            return

    console.print()
    console.print(Rule(f"[bold]Create Agent: {name}", style="cyan"))
    console.print()

    # Description
    description = Prompt.ask("  Description", default=f"{name} agent")

    # Provider selection
    llm = cfg.get("llm", {})
    primary = llm.get("primary", {})
    if not primary and "provider" in llm:
        primary = {"provider": llm["provider"], "model": llm.get("model", "")}

    available_providers = list(cfg.get("providers", {}).keys())
    console.print(f"  Available LLM providers: {', '.join(available_providers)}")
    if primary:
        console.print(f"  Primary: {primary.get('provider', '?')} / {primary.get('model', '?')}")

    use_primary = Confirm.ask("  Use primary LLM for this agent?", default=True)

    provider = ""
    model = ""
    if not use_primary:
        console.print("  Choose a provider for this agent:")
        for i, pid in enumerate(available_providers, 1):
            pmodel = cfg.get("providers", {}).get(pid, {}).get("model", "?")
            console.print(f"    [cyan]{i}.[/] {pid} ({pmodel})")
        choice = Prompt.ask("  Select", default="1")
        idx = int(choice) - 1 if choice.isdigit() else 0
        idx = max(0, min(idx, len(available_providers) - 1))
        provider = available_providers[idx]
        model = cfg.get("providers", {}).get(provider, {}).get("model", "")

    # Workspace
    workspace_path = str(config.PROJECT_ROOT / "workspaces" / name)
    console.print(f"  Workspace: {workspace_path}")

    # Bootstrap a temporary registry to generate TOOLS.md with builtin + shared tools
    from core.tools import ToolRegistry, register_builtin_tools
    from core.tool_discovery import discover_shared_tools
    temp_registry = ToolRegistry()
    register_builtin_tools(temp_registry)
    discover_shared_tools(temp_registry)

    # Create workspace with tools/ subdir and auto-generated TOOLS.md
    created_files = create_workspace(workspace_path, name, registry=temp_registry)
    if created_files:
        console.print(f"\n  [green]Created workspace files:[/]")
        for f in created_files:
            console.print(f"    {f}")
    console.print(f"    {workspace_path}/tools/  [dim](custom tools dir)[/]")

    # Save to alfred.json
    if "agents" not in cfg:
        cfg["agents"] = {}

    cfg["agents"][name] = {
        "workspace": f"workspaces/{name}",
        "description": description,
        "status": "active",
    }
    if provider:
        cfg["agents"][name]["provider"] = provider
    if model:
        cfg["agents"][name]["model"] = model

    _save_config(cfg)

    console.print(f"\n  [bold green]Agent '{name}' created![/]")
    console.print(f"  Chat with it: [bold]alfred agent chat {name}[/]")
    console.print(f"  Edit workspace: [bold]{workspace_path}/[/]")
    console.print()


def cmd_agent_list():
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config
    from core.scheduler import describe_cron

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    agents = cfg.get("agents", {})

    if not agents:
        console.print("\n  No agents configured.")
        console.print("  Create one: [bold]alfred agent create <name>[/]\n")
        return

    console.print()
    table = Table(box=box.ROUNDED, title="Agents", title_style="bold cyan")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Status")
    table.add_column("Provider / Model")
    table.add_column("Schedules")

    default_llm = cfg.get("llm", {})

    for name, acfg in agents.items():
        # Status
        status = acfg.get("status", "active")
        if status == "paused":
            status_str = "[yellow]paused[/]"
        else:
            status_str = "[green]active[/]"

        # Provider
        provider = acfg.get("provider", default_llm.get("provider", "?"))
        model = acfg.get("model", default_llm.get("model", "?"))
        # Truncate model name for display
        if len(model) > 25:
            model = model[:22] + "..."
        llm_str = f"{provider}/{model}"

        # Schedules
        schedules = acfg.get("schedules", [])
        if schedules:
            enabled = sum(1 for s in schedules if s.get("enabled", True))
            total = len(schedules)
            sched_str = f"{enabled}/{total} active"
        else:
            sched_str = "[dim]none[/]"

        table.add_row(
            name,
            acfg.get("description", ""),
            status_str,
            llm_str,
            sched_str,
        )

    console.print(table)
    console.print()


def cmd_agent_info(name: str):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from core.config import CONFIG_FILE, _load_config, config
    from core.scheduler import describe_cron
    from pathlib import Path

    console = Console()

    cfg = _load_config()
    agents = cfg.get("agents", {})

    if name not in agents:
        console.print(f"\n  [red]Agent '{name}' not found.[/]")
        if agents:
            console.print(f"  Available: {', '.join(agents.keys())}")
        console.print()
        return

    acfg = agents[name]
    default_llm = cfg.get("llm", {})
    console.print()

    # Agent details table
    table = Table(box=box.ROUNDED, title=f"Agent: {name}", title_style="bold cyan", show_header=False)
    table.add_column("Key", style="bold", width=18)
    table.add_column("Value")

    table.add_row("Description", acfg.get("description", "(none)"))
    table.add_row("Status", "[yellow]paused[/]" if acfg.get("status") == "paused" else "[green]active[/]")

    # LLM
    provider = acfg.get("provider", default_llm.get("provider", "?"))
    model = acfg.get("model", default_llm.get("model", "?"))
    is_custom = bool(acfg.get("provider"))
    llm_label = f"{provider} / {model}"
    if not is_custom:
        llm_label += " [dim](from primary)[/]"
    table.add_row("LLM", llm_label)

    # Workspace
    workspace = acfg.get("workspace", "?")
    if not Path(workspace).is_absolute():
        workspace_full = str(config.PROJECT_ROOT / workspace)
    else:
        workspace_full = workspace
    table.add_row("Workspace", workspace_full)

    # Workspace files
    ws_path = Path(workspace_full)
    if ws_path.exists():
        files = sorted(ws_path.glob("*.md"))
        file_list = ", ".join(f.name for f in files) if files else "(empty)"
        table.add_row("Workspace Files", file_list)

    # Memory
    memory_enabled = acfg.get("memory_enabled", True)
    table.add_row("Memory", "[green]enabled[/]" if memory_enabled else "[dim]disabled[/]")

    console.print(table)

    # Schedules
    schedules = acfg.get("schedules", [])
    if schedules:
        from core.scheduler import Schedule as _Schedule, next_run as _next_run
        console.print()
        sched_table = Table(box=box.ROUNDED, title="Schedules", title_style="bold")
        sched_table.add_column("ID", style="cyan", width=10)
        sched_table.add_column("Schedule")
        sched_table.add_column("Task")
        sched_table.add_column("Status")
        sched_table.add_column("Runs", justify="right")
        sched_table.add_column("Last Run")

        for s in schedules:
            sched = _Schedule.from_dict(s)
            task = sched.task
            if len(task) > 40:
                task = task[:37] + "..."

            status = "[green]active[/]" if sched.enabled else "[yellow]paused[/]"
            if sched.max_retries > 0:
                status += f" [dim]+{sched.max_retries}r[/]"

            # Run stats
            if sched.run_count > 0:
                rate = sched.success_rate
                rate_color = "green" if rate >= 90 else "yellow" if rate >= 50 else "red"
                runs_display = f"{sched.run_count} [{rate_color}]({rate:.0f}%)[/]"
            else:
                runs_display = "[dim]0[/]"

            if sched.last_run:
                run_display = sched.last_run[:16].replace("T", " ")
                if sched.last_result and "error" in sched.last_result:
                    run_display += f" [red]\u2717[/]"
            else:
                run_display = "[dim]never[/]"

            sched_table.add_row(sched.id, describe_cron(sched.cron), task, status, runs_display, run_display)

        console.print(sched_table)

    console.print()
    console.print(f"  [dim]Chat:[/]      alfred agent chat {name}")
    console.print(f"  [dim]Pause:[/]     alfred agent pause {name}")
    console.print(f"  [dim]Schedule:[/]  alfred agent schedule add {name}")
    console.print()


def cmd_agent_pause(name: str):
    from rich.console import Console
    from core.config import CONFIG_FILE, _load_config, _save_config

    console = Console()
    cfg = _load_config()

    if name not in cfg.get("agents", {}):
        console.print(f"\n  [red]Agent '{name}' not found.[/]\n")
        return

    if cfg["agents"][name].get("status") == "paused":
        console.print(f"\n  [yellow]Agent '{name}' is already paused.[/]\n")
        return

    cfg["agents"][name]["status"] = "paused"
    _save_config(cfg)

    # Count disabled schedules
    schedules = cfg["agents"][name].get("schedules", [])
    sched_msg = f" ({len(schedules)} schedule(s) suspended)" if schedules else ""

    console.print(f"\n  [yellow]Agent '{name}' paused.{sched_msg}[/]")
    console.print(f"  Resume: [bold]alfred agent resume {name}[/]\n")


def cmd_agent_resume(name: str):
    from rich.console import Console
    from core.config import CONFIG_FILE, _load_config, _save_config

    console = Console()
    cfg = _load_config()

    if name not in cfg.get("agents", {}):
        console.print(f"\n  [red]Agent '{name}' not found.[/]\n")
        return

    if cfg["agents"][name].get("status", "active") != "paused":
        console.print(f"\n  [dim]Agent '{name}' is already active.[/]\n")
        return

    cfg["agents"][name]["status"] = "active"
    _save_config(cfg)

    schedules = cfg["agents"][name].get("schedules", [])
    enabled = sum(1 for s in schedules if s.get("enabled", True))
    sched_msg = f" ({enabled} schedule(s) resumed)" if schedules else ""

    console.print(f"\n  [green]Agent '{name}' resumed.{sched_msg}[/]\n")


def cmd_agent_delete(name: str):
    from rich.console import Console
    from rich.prompt import Confirm
    from core.config import CONFIG_FILE, _load_config, _save_config, config
    from pathlib import Path
    import shutil

    console = Console()
    cfg = _load_config()

    if name not in cfg.get("agents", {}):
        console.print(f"\n  [red]Agent '{name}' not found.[/]\n")
        return

    acfg = cfg["agents"][name]
    workspace = acfg.get("workspace", "")
    schedules = acfg.get("schedules", [])

    console.print(f"\n  [bold red]Delete agent '{name}'?[/]")
    console.print(f"  Workspace: {workspace}")
    if schedules:
        console.print(f"  Schedules: {len(schedules)}")

    if not Confirm.ask("\n  This removes the agent from alfred.json. Continue?", default=False):
        console.print("  [dim]Cancelled.[/]\n")
        return

    # Remove from config
    del cfg["agents"][name]
    _save_config(cfg)
    console.print(f"  [green]Removed '{name}' from config.[/]")

    # Optionally delete workspace
    if workspace:
        ws_path = Path(workspace)
        if not ws_path.is_absolute():
            ws_path = config.PROJECT_ROOT / workspace
        if ws_path.exists():
            delete_ws = Confirm.ask(f"  Also delete workspace at {ws_path}?", default=False)
            if delete_ws:
                shutil.rmtree(ws_path)
                console.print(f"  [green]Workspace deleted.[/]")
            else:
                console.print(f"  [dim]Workspace preserved at {ws_path}[/]")

    console.print()


def cmd_agent_chat(name: str, session_id: str = None):
    from rich.console import Console
    from rich.panel import Panel
    from core.config import CONFIG_FILE, _load_config
    from core.agent import AgentManager, Agent

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    agents = cfg.get("agents", {})

    if name not in agents:
        console.print(f"\n  [red]Agent '{name}' not found.[/]")
        if agents:
            console.print(f"  Available: {', '.join(agents.keys())}")
        console.print(f"  Create it: [bold]alfred agent create {name}[/]\n")
        return

    # Check if paused
    if agents[name].get("status") == "paused":
        console.print(f"\n  [yellow]Agent '{name}' is paused. Resume first:[/]")
        console.print(f"  [bold]alfred agent resume {name}[/]\n")
        return

    # Load agent with optional named session
    manager = AgentManager()
    agent_config = manager._configs[name]

    # Ensure workspace exists
    from pathlib import Path
    workspace = Path(agent_config.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "memory").mkdir(exist_ok=True)
    (workspace / "tools").mkdir(exist_ok=True)

    agent = Agent(agent_config, session_id=session_id)

    console.print()
    session = agent.session_info
    session_label = f"session: {session_id}" if session_id else "default session"
    session_status = ""
    if session["turns"] > 0:
        session_status = f" | [green]{session['turns']} turns restored[/]"
    console.print(Panel(
        f"[bold cyan]{name}[/] | {agent.llm.provider}/{agent.llm.model} | "
        f"{len(agent._get_available_tools())} tools | {session_label}{session_status}\n"
        f"[dim]'quit' to exit | 'reset' to clear | 'approve <cmd>' to allow a command[/]",
        title="Alfred Agent Chat",
        border_style="cyan",
    ))
    console.print()

    while True:
        try:
            user_input = console.input("[bold green]you>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]Goodbye.[/]\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("  [dim]Goodbye.[/]\n")
            break
        if user_input.lower() == "reset":
            agent.reset()
            console.print("  [dim]Session reset.[/]\n")
            continue

        # Handle command approval: "approve <command>"
        if user_input.lower().startswith("approve "):
            cmd_to_approve = user_input[8:].strip()
            if cmd_to_approve:
                result = agent.registry.execute("run_command_approve", {"command_name": cmd_to_approve})
                console.print(f"  [dim]{result}[/]\n")
            else:
                console.print("  [dim]Usage: approve <command_name>[/]\n")
            continue

        # Run agent
        console.print()
        try:
            response = agent.run(user_input)
            console.print(f"[bold cyan]{name}>[/] {response}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
        console.print()


# ─── Schedule Commands ───────────────────────────────────────────

def cmd_agent_schedule_add(name: str):
    from rich.console import Console
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.rule import Rule
    from core.config import CONFIG_FILE, _load_config
    from core.scheduler import add_schedule, describe_cron, cron_matches, next_run

    console = Console()
    cfg = _load_config()

    if name not in cfg.get("agents", {}):
        console.print(f"\n  [red]Agent '{name}' not found.[/]\n")
        return

    console.print()
    console.print(Rule(f"[bold]Add Schedule: {name}", style="cyan"))
    console.print()

    # Cron expression
    console.print("  [dim]Cron format: minute hour day_of_month month day_of_week[/]")
    console.print("  [dim]Examples:[/]")
    console.print("    [dim]30 9 * * 1-5     = 09:30 weekdays[/]")
    console.print("    [dim]0 16 * * 1-5     = 16:00 weekdays[/]")
    console.print("    [dim]*/15 * * * *     = every 15 minutes[/]")
    console.print("    [dim]0 8 * * *        = 08:00 every day[/]")
    console.print()

    cron = Prompt.ask("  Cron expression")
    human = describe_cron(cron)
    console.print(f"  [dim]Parsed: {human}[/]")

    # Show next fire time
    nxt = next_run(cron)
    if nxt:
        console.print(f"  [dim]Next run: {nxt.strftime('%Y-%m-%d %H:%M')}[/]")

    # Task description
    console.print()
    task = Prompt.ask("  Task (what should the agent do?)")

    # Retry settings (optional)
    retries = 0
    retry_delay = 30
    if Confirm.ask("\n  Enable retries on failure?", default=False):
        retries = IntPrompt.ask("  Max retries (1-3)", default=1)
        retries = max(0, min(retries, 3))
        retry_delay = IntPrompt.ask("  Retry delay (seconds)", default=30)
        retry_delay = max(5, retry_delay)

    schedule = add_schedule(name, cron, task)

    # Update retry settings if configured
    if retries > 0:
        from core.scheduler import update_schedule_retries
        update_schedule_retries(name, schedule.id, retries, retry_delay)

    console.print(f"\n  [green]Schedule added![/]")
    console.print(f"  ID: [cyan]{schedule.id}[/]")
    console.print(f"  When: {human}")
    console.print(f"  Task: {task}")
    if retries > 0:
        console.print(f"  Retries: {retries} (delay: {retry_delay}s)")
    console.print()


def cmd_agent_schedule_list(name: str = None):
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config
    from core.scheduler import describe_cron, next_run, Schedule

    console = Console()
    cfg = _load_config()

    console.print()

    # Determine which agents to show
    if name:
        if name not in cfg.get("agents", {}):
            console.print(f"  [red]Agent '{name}' not found.[/]\n")
            return
        agents_to_show = {name: cfg["agents"][name]}
    else:
        agents_to_show = cfg.get("agents", {})

    if not agents_to_show:
        console.print("  No agents configured.\n")
        return

    has_any = False

    for agent_name, acfg in agents_to_show.items():
        schedules = acfg.get("schedules", [])
        if not schedules:
            continue

        has_any = True
        agent_status = acfg.get("status", "active")
        title = f"{agent_name}"
        if agent_status == "paused":
            title += " [yellow](paused)[/]"

        table = Table(box=box.ROUNDED, title=title, title_style="bold cyan")
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Schedule")
        table.add_column("Task")
        table.add_column("Status")
        table.add_column("Runs", justify="right")
        table.add_column("Last Run")
        table.add_column("Next Run")

        for s in schedules:
            sched = Schedule.from_dict(s)
            task = sched.task
            if len(task) > 40:
                task = task[:37] + "..."

            status = "[green]active[/]" if sched.enabled else "[yellow]paused[/]"
            if agent_status == "paused":
                status = "[yellow]agent paused[/]"

            # Run stats
            if sched.run_count > 0:
                rate = sched.success_rate
                rate_color = "green" if rate >= 90 else "yellow" if rate >= 50 else "red"
                runs_display = f"{sched.run_count} [{rate_color}]({rate:.0f}%)[/]"
            else:
                runs_display = "[dim]0[/]"

            # Retry badge
            if sched.max_retries > 0:
                status += f" [dim]+{sched.max_retries}r[/]"

            # Consecutive failure warning
            if sched.consecutive_failures >= 3:
                status += f" [red]({sched.consecutive_failures} fails)[/]"

            run_display = sched.last_run[:16].replace("T", " ") if sched.last_run else "[dim]never[/]"

            # Next run
            nxt = next_run(sched.cron) if sched.enabled else None
            if nxt:
                from datetime import datetime as _dt
                delta = nxt - _dt.now()
                total_mins = int(delta.total_seconds() / 60)
                if total_mins < 60:
                    next_display = f"{total_mins}m"
                elif total_mins < 1440:
                    next_display = f"{total_mins // 60}h {total_mins % 60}m"
                else:
                    next_display = f"{total_mins // 1440}d {(total_mins % 1440) // 60}h"
                next_display = f"[green]{next_display}[/]"
            else:
                next_display = "[dim]—[/]"

            table.add_row(sched.id, describe_cron(sched.cron), task, status, runs_display, run_display, next_display)

        console.print(table)
        console.print()

    if not has_any:
        console.print("  No schedules configured.")
        console.print("  Add one: [bold]alfred agent schedule add <name>[/]\n")


def cmd_agent_schedule_remove(name: str, schedule_id: str):
    from rich.console import Console
    from rich.prompt import Confirm
    from core.config import CONFIG_FILE, _load_config
    from core.scheduler import remove_schedule

    console = Console()
    cfg = _load_config()

    if name not in cfg.get("agents", {}):
        console.print(f"\n  [red]Agent '{name}' not found.[/]\n")
        return

    # Find the schedule to show details
    schedules = cfg["agents"][name].get("schedules", [])
    target = next((s for s in schedules if s.get("id") == schedule_id), None)

    if not target:
        console.print(f"\n  [red]Schedule '{schedule_id}' not found for agent '{name}'.[/]")
        console.print(f"  Run: [bold]alfred agent schedule list {name}[/]\n")
        return

    console.print(f"\n  Remove schedule [cyan]{schedule_id}[/]?")
    console.print(f"  Task: {target.get('task', '?')}")
    console.print(f"  Cron: {target.get('cron', '?')}")

    if not Confirm.ask("\n  Confirm?", default=True):
        console.print("  [dim]Cancelled.[/]\n")
        return

    if remove_schedule(name, schedule_id):
        console.print(f"\n  [green]Schedule removed.[/]\n")
    else:
        console.print(f"\n  [red]Failed to remove schedule.[/]\n")


def cmd_agent_schedule_enable(name: str, schedule_id: str):
    """Enable a paused schedule."""
    from rich.console import Console
    from core.config import _load_config
    from core.scheduler import toggle_schedule

    console = Console()
    if toggle_schedule(name, schedule_id, enabled=True):
        console.print(f"\n  [green]\u2713[/] Schedule [cyan]{schedule_id}[/] enabled.\n")
    else:
        console.print(f"\n  [red]Schedule '{schedule_id}' not found for agent '{name}'.[/]\n")


def cmd_agent_schedule_disable(name: str, schedule_id: str):
    """Disable a schedule (pauses it without removing)."""
    from rich.console import Console
    from core.scheduler import toggle_schedule

    console = Console()
    if toggle_schedule(name, schedule_id, enabled=False):
        console.print(f"\n  [yellow]\u2713[/] Schedule [cyan]{schedule_id}[/] disabled.\n")
    else:
        console.print(f"\n  [red]Schedule '{schedule_id}' not found for agent '{name}'.[/]\n")


def cmd_agent_schedule_run(name: str, schedule_id: str):
    """Manually trigger a scheduled task right now."""
    from rich.console import Console
    from core.config import CONFIG_FILE, _load_config
    from core.scheduler import get_schedule, run_schedule_now
    from core.agent import Agent, AgentConfig
    from pathlib import Path
    from core.config import config as _config

    console = Console()

    schedule = get_schedule(name, schedule_id)
    if not schedule:
        console.print(f"\n  [red]Schedule '{schedule_id}' not found for agent '{name}'.[/]\n")
        return

    console.print(f"\n  Running [cyan]{schedule_id}[/]: {schedule.task[:60]}")
    console.print(f"  [dim]This runs synchronously — wait for it to complete...[/]\n")

    def _runner(agent_name: str, task: str) -> str:
        cfg = _load_config()
        agent_data = dict(cfg.get("agents", {}).get(agent_name, {}))
        agent_data["name"] = agent_name

        workspace = Path(agent_data.get("workspace", f"workspaces/{agent_name}"))
        if not workspace.is_absolute():
            workspace = _config.PROJECT_ROOT / workspace
        agent_data["workspace"] = str(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "memory").mkdir(exist_ok=True)
        (workspace / "tools").mkdir(exist_ok=True)

        agent_config = AgentConfig.from_dict(agent_data)
        agent = Agent(agent_config, session_id="schedule")
        return agent.run(task)

    try:
        result = run_schedule_now(name, schedule_id, agent_runner=_runner)
        console.print(f"  [green]\u2713[/] Completed!")
        console.print(f"\n  [bold cyan]{name}>[/] {result}\n")
    except Exception as e:
        console.print(f"  [red]\u2717 Failed: {e}[/]\n")


def cmd_agent_schedule_history(name: str, schedule_id: str):
    """Show run history for a schedule."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.scheduler import get_schedule, get_schedule_history

    console = Console()

    schedule = get_schedule(name, schedule_id)
    if not schedule:
        console.print(f"\n  [red]Schedule '{schedule_id}' not found for agent '{name}'.[/]\n")
        return

    console.print()
    console.print(f"  [bold cyan]{schedule_id}[/] — {schedule.task[:60]}")
    console.print(f"  Cron: [dim]{schedule.cron}[/]")

    # Stats summary
    if schedule.run_count > 0:
        rate = schedule.success_rate
        rate_color = "green" if rate >= 90 else "yellow" if rate >= 50 else "red"
        console.print(
            f"  Runs: {schedule.run_count} | "
            f"[green]{schedule.success_count}[/] success, "
            f"[red]{schedule.fail_count}[/] failed | "
            f"Rate: [{rate_color}]{rate:.0f}%[/]"
        )
        if schedule.consecutive_failures > 0:
            console.print(f"  [red]Consecutive failures: {schedule.consecutive_failures}[/]")
    else:
        console.print("  [dim]No runs yet.[/]")

    history = get_schedule_history(name, schedule_id)
    if not history:
        console.print("\n  [dim]No run history.[/]\n")
        return

    console.print()
    table = Table(box=box.ROUNDED, title="Run History", title_style="bold")
    table.add_column("Time", style="dim")
    table.add_column("Result")
    table.add_column("Duration", justify="right")
    table.add_column("Flags", style="dim")

    for run in history:
        ts = run.timestamp[:19].replace("T", " ") if run.timestamp else "?"

        if run.result == "success":
            result_display = "[green]success[/]"
        elif "will retry" in run.result:
            result_display = f"[yellow]{run.result[:40]}[/]"
        else:
            result_display = f"[red]{run.result[:40]}[/]"

        duration = f"{run.elapsed_ms}ms" if run.elapsed_ms else "[dim]—[/]"

        flags = []
        if run.is_catchup:
            flags.append("catchup")
        if run.is_retry:
            flags.append("retry")
        flags_display = ", ".join(flags) if flags else ""

        table.add_row(ts, result_display, duration, flags_display)

    console.print(table)
    console.print()


def cmd_agent_schedule_retry(name: str, schedule_id: str, max_retries: int = None, delay: int = None):
    """Configure retry settings for a schedule."""
    from rich.console import Console
    from rich.prompt import IntPrompt
    from core.scheduler import get_schedule, update_schedule_retries

    console = Console()

    schedule = get_schedule(name, schedule_id)
    if not schedule:
        console.print(f"\n  [red]Schedule '{schedule_id}' not found for agent '{name}'.[/]\n")
        return

    console.print(f"\n  [bold cyan]{schedule_id}[/] — {schedule.task[:60]}")
    console.print(f"  Current: max_retries={schedule.max_retries}, delay={schedule.retry_delay_seconds}s")

    if max_retries is None:
        max_retries = IntPrompt.ask("\n  Max retries (0-3, 0=disabled)", default=schedule.max_retries)
    if delay is None:
        delay = IntPrompt.ask("  Retry delay (seconds)", default=schedule.retry_delay_seconds)

    max_retries = max(0, min(max_retries, 3))
    delay = max(5, delay)

    if update_schedule_retries(name, schedule_id, max_retries, delay):
        if max_retries > 0:
            console.print(f"\n  [green]\u2713[/] Retries set: {max_retries} attempts, {delay}s delay.\n")
        else:
            console.print(f"\n  [green]\u2713[/] Retries disabled.\n")
    else:
        console.print(f"\n  [red]Failed to update.[/]\n")


# ─── Models Commands ─────────────────────────────────────────────

def cmd_models_update(provider_id: str = None):
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config
    from core.models import update_models

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    console.print()

    if provider_id:
        console.print(f"  [dim]Fetching models from {provider_id}...[/]")
    else:
        console.print("  [dim]Fetching models from all configured providers...[/]")

    results = update_models(provider_id)

    for pid, data in results.items():
        if isinstance(data, dict) and "error" in data:
            console.print(f"\n  [red]{pid}: {data['error']}[/]")
        elif isinstance(data, list):
            console.print(f"\n  [green]{pid}:[/] {len(data)} models found")

            table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
            table.add_column("#", style="dim", width=4)
            table.add_column("Model ID", style="cyan")
            table.add_column("Name")

            for i, m in enumerate(data[:20], 1):  # Show top 20
                model_id = m.get("id", "?")
                model_name = m.get("name", "")
                if model_name == model_id:
                    model_name = ""
                table.add_row(str(i), model_id, model_name)

            if len(data) > 20:
                table.add_row("", f"... and {len(data) - 20} more", "")

            console.print(table)
        else:
            console.print(f"\n  [yellow]{pid}: unexpected response[/]")

    console.print()
    console.print("  [dim]Models cached in data/models_cache.json[/]")
    console.print("  [dim]View anytime: alfred models list[/]")
    console.print()


def cmd_models_list(provider_id: str = None):
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.models import get_cached_models

    console = Console()

    cache = get_cached_models(provider_id)

    if not cache or all(not v for v in cache.values()):
        console.print("\n  [yellow]No cached models. Run: alfred models update[/]\n")
        return

    console.print()

    for pid, pdata in cache.items():
        if not pdata:
            continue

        models = pdata.get("models", [])
        updated = pdata.get("updated_at", "?")

        console.print(f"  [bold]{pid}[/] [dim](updated: {updated})[/]")

        if not models:
            console.print("    (no models)")
            continue

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("#", style="dim", width=4)
        table.add_column("Model ID", style="cyan")

        for i, m in enumerate(models[:25], 1):
            table.add_row(str(i), m.get("id", "?"))

        if len(models) > 25:
            table.add_row("", f"... and {len(models) - 25} more")

        console.print(table)
        console.print()

    console.print("  [dim]Refresh: alfred models update[/]\n")


# ─── Provider Commands ───────────────────────────────────────────

def cmd_provider_add(provider_id: str):
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.rule import Rule
    from core.config import CONFIG_FILE, _load_config, _save_config
    from core.llm import LLMClient, detect_ollama
    from core.models import get_cached_models

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    # ── Brave Search (special case — search API, not LLM) ──
    if provider_id == "brave":
        cfg = _load_config()
        _cmd_provider_add_brave(console, cfg)
        return

    PROVIDERS = {
        "anthropic": {
            "name": "Anthropic (Claude)",
            "key_hint": "Starts with sk-ant-...",
            "models": [
                "claude-sonnet-4-6",
                "claude-opus-4-6",
                "claude-haiku-4-5-20251001",
                "claude-sonnet-4-5-20250929",
                "claude-opus-4-5-20251101",
                "claude-opus-4-20250514",
                "claude-haiku-3-5-20241022",
            ],
            "default_model": "claude-sonnet-4-6",
        },
        "xai": {
            "name": "xAI (Grok)",
            "key_hint": "Starts with xai-...",
            "models": [
                "grok-4-1-fast-reasoning",
                "grok-4-1-fast-non-reasoning",
                "grok-code-fast-1",
                "grok-4",
                "grok-4-fast-reasoning",
                "grok-4-fast-non-reasoning",
                "grok-3",
                "grok-3-mini",
            ],
            "default_model": "grok-4-1-fast-reasoning",
        },
        "openai": {
            "name": "OpenAI (GPT)",
            "key_hint": "Starts with sk-...",
            "models": [
                "gpt-5.2",
                "gpt-5.2-codex",
                "gpt-5.1",
                "gpt-5.1-codex",
                "gpt-4.1-2025-04-14",
                "gpt-4.1-mini-2025-04-14",
                "o3-2025-04-16",
                "o4-mini-2025-04-16",
            ],
            "default_model": "gpt-5.2",
        },
        "ollama": {
            "name": "Ollama (Local)",
            "key_hint": "No key needed",
            "models": [],
            "default_model": "llama3.1",
        },
    }

    if provider_id not in PROVIDERS:
        console.print(f"\n  [red]Unknown provider '{provider_id}'[/]")
        all_providers = list(PROVIDERS.keys()) + ["brave"]
        console.print(f"  Available: {', '.join(all_providers)}\n")
        return

    prov = PROVIDERS[provider_id]
    cfg = _load_config()
    console.print()
    console.print(Rule(f"[bold]Add Provider: {prov['name']}", style="cyan"))
    console.print()

    provider_entry = {}

    # API key
    if provider_id != "ollama":
        # Check if key already exists
        existing = cfg.get("providers", {}).get(provider_id, {}).get("api_key", "")
        if existing:
            masked = existing[:8] + "..." + existing[-4:]
            console.print(f"  Existing key: {masked}")

        key = Prompt.ask(f"  API key ({prov['key_hint']})")
        provider_entry["api_key"] = key.strip()
    else:
        running, models = detect_ollama()
        if running:
            console.print(f"  [green]Ollama running with {len(models)} model(s)[/]")
            prov["models"] = models
        else:
            console.print("  [yellow]Ollama not detected. Start with: ollama serve[/]")

    # Check if we have cached models from `alfred models update`
    cached = get_cached_models(provider_id)
    cached_models = cached.get(provider_id, {}).get("models", [])
    if cached_models and provider_id != "ollama":
        cached_ids = [m.get("id", "") for m in cached_models]
        console.print(f"  [dim]({len(cached_ids)} models available from cache — run 'alfred models update' to refresh)[/]")
        # Use cached models instead of hardcoded if available
        prov["models"] = cached_ids

    # Model selection
    models = prov["models"]
    if models:
        console.print("  Models:")
        for i, m in enumerate(models, 1):
            console.print(f"    [cyan]{i}.[/] {m}")
        choice = Prompt.ask("  Select model", default="1")
        idx = int(choice) - 1 if choice.isdigit() else 0
        idx = max(0, min(idx, len(models) - 1))
        provider_entry["model"] = models[idx]
    else:
        provider_entry["model"] = Prompt.ask("  Model name", default=prov["default_model"])

    # Test connection
    console.print(f"  [dim]Testing {prov['name']}...[/]", end="")
    client = LLMClient(
        provider=provider_id,
        api_key=provider_entry.get("api_key", ""),
        model=provider_entry["model"],
    )
    success, msg = client.test_connection()
    if success:
        console.print(f"\r  [green]\u2713 {msg}[/]")
    else:
        console.print(f"\r  [red]\u2717 {msg}[/]")

    # Save provider
    if "providers" not in cfg:
        cfg["providers"] = {}
    cfg["providers"][provider_id] = provider_entry

    console.print(f"\n  [bold green]{prov['name']} added![/]")

    # Migrate old flat llm format to primary/secondary if needed
    if "llm" not in cfg:
        cfg["llm"] = {}
    llm = cfg["llm"]
    if "provider" in llm and "primary" not in llm:
        # Migrate old format
        llm["primary"] = {"provider": llm.pop("provider"), "model": llm.pop("model", "")}

    # Ask about role: primary, secondary, or neither
    from rich.prompt import Confirm

    current_primary = llm.get("primary", {}).get("provider", "")
    current_secondary = llm.get("secondary", {}).get("provider", "")

    if current_primary:
        console.print(f"  Current primary: [bold]{current_primary}[/]")
    if current_secondary:
        console.print(f"  Current secondary: [bold]{current_secondary}[/]")

    console.print()
    role_choice = Prompt.ask(
        "  Set as",
        choices=["primary", "secondary", "skip"],
        default="secondary" if current_primary else "primary",
    )

    if role_choice == "primary":
        # If there was a primary, demote it to secondary
        if current_primary and current_primary != provider_id:
            old_primary = llm.get("primary", {})
            if old_primary:
                llm["secondary"] = old_primary
                console.print(f"  [dim]{current_primary} moved to secondary[/]")
        llm["primary"] = {"provider": provider_id, "model": provider_entry["model"]}
        console.print(f"  [green]\u2713 {prov['name']} set as primary[/]")

    elif role_choice == "secondary":
        llm["secondary"] = {"provider": provider_id, "model": provider_entry["model"]}
        console.print(f"  [green]\u2713 {prov['name']} set as secondary (fallback)[/]")

    _save_config(cfg)
    console.print()


def _cmd_provider_add_brave(console, cfg):
    """Add Brave Search API key. Search-only provider, no LLM."""
    from rich.prompt import Prompt
    from rich.rule import Rule
    from core.config import _save_config

    console.print()
    console.print(Rule("[bold]Add Provider: Brave Search", style="cyan"))
    console.print()
    console.print("  Brave Search API provides real web search results.")
    console.print("  Free tier: 2,000 queries/month. No credit card needed.")
    console.print("  Get a key at: [cyan]https://brave.com/search/api/[/]")
    console.print()

    # Check existing key
    existing = cfg.get("providers", {}).get("brave", {}).get("api_key", "")
    if existing:
        masked = existing[:8] + "..." + existing[-4:]
        console.print(f"  Existing key: {masked}")

    key = Prompt.ask("  API key").strip()
    if not key:
        console.print("  [yellow]No key provided, cancelled.[/]\n")
        return

    # Test the key with a simple search
    console.print("  [dim]Testing Brave Search...[/]", end="")
    import urllib.request
    import urllib.error
    import urllib.parse
    import json

    params = urllib.parse.urlencode({"q": "test", "count": 1})
    url = f"https://api.search.brave.com/res/v1/web/search?{params}"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": key,
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("web", {}).get("results", [])
            console.print(f"\r  [green]✓ Connected — {len(results)} result(s) returned[/]")
    except urllib.error.HTTPError as e:
        console.print(f"\r  [red]✗ HTTP {e.code} — check your API key[/]")
    except Exception as e:
        console.print(f"\r  [red]✗ {e}[/]")

    # Save
    if "providers" not in cfg:
        cfg["providers"] = {}
    cfg["providers"]["brave"] = {"api_key": key}

    _save_config(cfg)
    console.print(f"\n  [bold green]Brave Search added![/]")
    console.print("  Your agents can now use the web_search tool with real results.")
    console.print()


# ─── Service Commands ────────────────────────────────────────────

def cmd_service_list():
    """List all configured external services with masked keys."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    services = cfg.get("services", {})

    if not services:
        console.print("\n  [dim]No services configured.[/]")
        console.print("  Add one with: [cyan]alfred service add <name>[/]\n")
        return

    console.print()
    table = Table(
        title="Configured Services",
        box=box.ROUNDED,
        title_style="bold white",
    )
    table.add_column("Service", style="cyan bold")
    table.add_column("API Key", style="dim")
    table.add_column("Secret Key", style="dim")
    table.add_column("Domains", style="white")

    for name, svc_cfg in services.items():
        # Mask keys
        api_key = svc_cfg.get("api_key", "")
        secret_key = svc_cfg.get("secret_key", "")
        masked_api = (api_key[:4] + "..." + api_key[-4:]) if len(api_key) > 8 else ("***" if api_key else "—")
        masked_secret = (secret_key[:4] + "..." + secret_key[-4:]) if len(secret_key) > 8 else ("***" if secret_key else "—")

        # Collect domains
        domains = []
        for key, val in svc_cfg.items():
            if isinstance(val, str) and val.startswith("http"):
                try:
                    from urllib.parse import urlparse
                    host = urlparse(val).hostname
                    if host:
                        domains.append(host)
                except Exception:
                    pass

        table.add_row(name, masked_api, masked_secret, "\n".join(domains) if domains else "—")

    console.print(table)
    console.print()


def cmd_service_add(name: str):
    """Add or update credentials for an external service."""
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.rule import Rule
    from core.config import CONFIG_FILE, _load_config, _save_config

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    console.print()
    console.print(Rule(f"[bold]Add Service: {name}", style="cyan"))
    console.print()

    # Check if already exists
    existing = cfg.get("services", {}).get(name, {})
    if existing:
        masked = existing.get("api_key", "")
        if len(masked) > 8:
            masked = masked[:4] + "..." + masked[-4:]
        console.print(f"  [yellow]Service '{name}' already configured (key: {masked})[/]")
        overwrite = Prompt.ask("  Overwrite?", choices=["y", "n"], default="n")
        if overwrite != "y":
            console.print("  [dim]Cancelled.[/]\n")
            return
        console.print()

    service_entry = {}

    # API Key
    api_key = Prompt.ask("  API Key")
    if api_key.strip():
        service_entry["api_key"] = api_key.strip()

    # Secret Key (optional)
    secret_key = Prompt.ask("  Secret Key (optional, press Enter to skip)", default="")
    if secret_key.strip():
        service_entry["secret_key"] = secret_key.strip()

    # Base URL
    base_url = Prompt.ask("  Base URL (e.g. https://api.example.com)")
    if base_url.strip():
        service_entry["base_url"] = base_url.strip()

    # Additional data URL (optional)
    data_url = Prompt.ask("  Data URL (optional, press Enter to skip)", default="")
    if data_url.strip():
        service_entry["data_url"] = data_url.strip()

    if not service_entry:
        console.print("  [yellow]No credentials provided. Cancelled.[/]\n")
        return

    # Save
    if "services" not in cfg:
        cfg["services"] = {}
    cfg["services"][name] = service_entry
    _save_config(cfg)

    console.print(f"\n  [bold green]Service '{name}' added![/]")
    console.print(f"  Auth headers will be auto-injected for requests to configured domains.")

    # Remind about auth handler
    console.print(f"\n  [dim]Note: If this service needs custom auth headers, add a handler to[/]")
    console.print(f"  [dim]tools/http_request.py in _SERVICE_AUTH_MAP.[/]\n")


def cmd_service_remove(name: str):
    """Remove an external service configuration."""
    from rich.console import Console
    from rich.prompt import Prompt
    from core.config import CONFIG_FILE, _load_config, _save_config

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    services = cfg.get("services", {})

    if name not in services:
        console.print(f"\n  [red]Service '{name}' not found.[/]")
        if services:
            console.print(f"  Available: {', '.join(services.keys())}")
        console.print()
        return

    confirm = Prompt.ask(f"  Remove service '{name}'?", choices=["y", "n"], default="n")
    if confirm != "y":
        console.print("  [dim]Cancelled.[/]\n")
        return

    del cfg["services"][name]
    _save_config(cfg)
    console.print(f"\n  [green]Service '{name}' removed.[/]\n")


def cmd_demo():
    import demo
    demo.main()


# ─── Session Commands ────────────────────────────────────────────

def cmd_session_list(agent_name: str = None):
    """List all saved conversation sessions."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config
    from core.agent import Agent
    from pathlib import Path
    from core.config import config as cfg_module

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    agents = cfg.get("agents", {})

    if not agents:
        console.print("\n  [dim]No agents configured.[/]\n")
        return

    # If a specific agent is given, only show that one
    agent_names = [agent_name] if agent_name else list(agents.keys())

    found_any = False
    for name in agent_names:
        if name not in agents:
            console.print(f"\n  [red]Agent '{name}' not found.[/]")
            continue

        workspace = agents[name].get("workspace", f"workspaces/{name}")
        workspace_path = Path(workspace)
        if not workspace_path.is_absolute():
            workspace_path = cfg_module.PROJECT_ROOT / workspace_path

        sessions = Agent.list_sessions(str(workspace_path))

        if not sessions:
            if agent_name:  # Only show "no sessions" if user asked for a specific agent
                console.print(f"\n  [dim]No saved sessions for '{name}'.[/]\n")
            continue

        found_any = True
        console.print()
        table = Table(
            box=box.ROUNDED,
            title=f"Sessions — {name}",
            title_style="bold cyan",
        )
        table.add_column("Session ID", style="bold")
        table.add_column("Turns", justify="right")
        table.add_column("Started", style="dim")
        table.add_column("Last Active", style="dim")

        for s in sessions:
            # Format timestamps to be human-readable
            started = s.get("started_at", "")[:19].replace("T", " ") if s.get("started_at") else "?"
            last = s.get("last_activity", "")[:19].replace("T", " ") if s.get("last_activity") else "?"
            turns = str(s.get("turn_count", 0))
            sid = s["session_id"]

            # Color-code session types
            if sid == "cli":
                sid_display = "[green]cli[/]"
            elif sid == "api":
                sid_display = "[blue]api[/]"
            elif sid == "webhook":
                sid_display = "[magenta]webhook[/]"
            elif sid.isdigit():
                sid_display = f"[yellow]discord:{sid[-6:]}[/]"
            else:
                sid_display = f"[cyan]{sid}[/]"

            error = s.get("error", "")
            if error:
                turns = f"[red]?[/]"
                sid_display += " [red](corrupt)[/]"

            table.add_row(sid_display, turns, started, last)

        console.print(table)

    if not found_any and not agent_name:
        console.print("\n  [dim]No saved sessions found.[/]")

    console.print()


def cmd_session_view(agent_name: str, session_id: str, last_n: int = None):
    """View conversation history from a saved session."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from core.config import CONFIG_FILE, _load_config
    from core.agent import Agent
    from pathlib import Path
    from core.config import config as cfg_module

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    agents = cfg.get("agents", {})

    if agent_name not in agents:
        console.print(f"\n  [red]Agent '{agent_name}' not found.[/]")
        return

    workspace = agents[agent_name].get("workspace", f"workspaces/{agent_name}")
    workspace_path = Path(workspace)
    if not workspace_path.is_absolute():
        workspace_path = cfg_module.PROJECT_ROOT / workspace_path

    messages = Agent.get_session_messages(str(workspace_path), session_id)

    if not messages:
        console.print(f"\n  [dim]No messages found for session '{session_id}'.[/]\n")
        return

    # Optionally show only last N turns
    if last_n:
        messages = messages[-(last_n * 2):]

    console.print()
    console.print(Panel(
        f"[bold cyan]{agent_name}[/] | session: [bold]{session_id}[/] | "
        f"{len(messages) // 2} turns shown",
        title="Session History",
        border_style="cyan",
    ))
    console.print()

    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")

        if role == "user":
            console.print(f"[bold green]you>[/] {content}")
        elif role == "assistant":
            console.print(f"[bold cyan]{agent_name}>[/] {content}")
        console.print()


def cmd_session_export(agent_name: str, session_id: str, output_path: str = None, format: str = "markdown"):
    """Export a session to a file."""
    from rich.console import Console
    from core.config import CONFIG_FILE, _load_config
    from core.agent import Agent
    from pathlib import Path
    from core.config import config as cfg_module

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    agents = cfg.get("agents", {})

    if agent_name not in agents:
        console.print(f"\n  [red]Agent '{agent_name}' not found.[/]")
        return

    workspace = agents[agent_name].get("workspace", f"workspaces/{agent_name}")
    workspace_path = Path(workspace)
    if not workspace_path.is_absolute():
        workspace_path = cfg_module.PROJECT_ROOT / workspace_path

    content = Agent.export_session(str(workspace_path), session_id, format=format)

    if not content:
        console.print(f"\n  [dim]No messages found for session '{session_id}'.[/]\n")
        return

    if output_path:
        Path(output_path).write_text(content)
        console.print(f"\n  [green]\u2713[/] Exported to {output_path}\n")
    else:
        # Print to stdout
        console.print(content)


def cmd_session_delete(agent_name: str, session_id: str):
    """Delete a saved session."""
    from rich.console import Console
    from rich.prompt import Confirm
    from core.config import CONFIG_FILE, _load_config
    from core.agent import Agent
    from pathlib import Path
    from core.config import config as cfg_module

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    agents = cfg.get("agents", {})

    if agent_name not in agents:
        console.print(f"\n  [red]Agent '{agent_name}' not found.[/]")
        return

    workspace = agents[agent_name].get("workspace", f"workspaces/{agent_name}")
    workspace_path = Path(workspace)
    if not workspace_path.is_absolute():
        workspace_path = cfg_module.PROJECT_ROOT / workspace_path

    # Check it exists first
    messages = Agent.get_session_messages(str(workspace_path), session_id)
    if not messages:
        console.print(f"\n  [dim]Session '{session_id}' not found or empty.[/]\n")
        return

    turns = len(messages) // 2
    if not Confirm.ask(f"\n  Delete session '{session_id}' ({turns} turns)?", default=False):
        console.print("  [dim]Cancelled.[/]\n")
        return

    deleted = Agent.delete_session(str(workspace_path), session_id)
    if deleted:
        console.print(f"  [green]\u2713[/] Session '{session_id}' deleted.\n")
    else:
        console.print(f"  [red]Could not delete session.[/]\n")


# ─── Tools Commands ──────────────────────────────────────────────

def cmd_tools_list(agent_name: str = None):
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.tools import ToolRegistry, register_builtin_tools
    from core.tool_discovery import discover_shared_tools

    console = Console()

    if agent_name:
        # Agent-specific: load full agent (includes workspace tools + meta-tools)
        from core.config import CONFIG_FILE, _load_config
        cfg = _load_config()
        agents = cfg.get("agents", {})

        if agent_name not in agents:
            console.print(f"\n  [red]Agent '{agent_name}' not found.[/]")
            if agents:
                console.print(f"  Available: {', '.join(agents.keys())}")
            console.print()
            return

        from core.agent import AgentManager
        manager = AgentManager()
        agent = manager.get(agent_name)
        registry = agent.registry
        available = agent._get_available_tools()
        title = f"Tools: {agent_name}"
    else:
        # Global view: builtins + shared tools only
        registry = ToolRegistry()
        register_builtin_tools(registry)
        shared = discover_shared_tools(registry)
        available = registry.names()
        title = "Tools (Global)"

    tools = registry.get_many(available)

    if not tools:
        console.print("\n  No tools found.\n")
        return

    console.print()
    table = Table(box=box.ROUNDED, title=title, title_style="bold cyan")
    table.add_column("Name", style="bold cyan")
    table.add_column("Category")
    table.add_column("Source")
    table.add_column("Description")

    # Group by source for cleaner display
    for source_label in ["builtin", "shared", "workspace"]:
        source_tools = [t for t in tools if t.source == source_label]
        for tool in source_tools:
            desc = tool.description
            if len(desc) > 60:
                desc = desc[:57] + "..."

            source_style = {
                "builtin": "[dim]builtin[/]",
                "shared": "[green]shared[/]",
                "workspace": "[yellow]workspace[/]",
            }.get(tool.source, tool.source)

            table.add_row(
                tool.name,
                tool.category or "general",
                source_style,
                desc,
            )

    console.print(table)
    console.print()

    # Summary
    by_source = {}
    for t in tools:
        by_source[t.source] = by_source.get(t.source, 0) + 1
    summary = ", ".join(f"{count} {source}" for source, count in sorted(by_source.items()))
    console.print(f"  [dim]{len(tools)} tools total ({summary})[/]")

    if not agent_name:
        console.print(f"  [dim]Agent-specific view: alfred tools list <agent_name>[/]")
    console.print()


# ─── Discord Commands ─────────────────────────────────────────────

def _discord_discover(token: str, guild_id: str = None) -> dict:
    """
    Briefly connect to Discord to discover guilds and optionally channels.
    Single connection, clean shutdown.

    Returns {"guilds": [{id, name}, ...], "channels": [{id, name, category}, ...]}
    """
    import asyncio
    import warnings
    import discord

    result = {"guilds": [], "channels": []}

    async def _discover():
        intents = discord.Intents.default()
        intents.guilds = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            try:
                for g in client.guilds:
                    result["guilds"].append({"id": str(g.id), "name": g.name})

                if guild_id:
                    guild = client.get_guild(int(guild_id))
                    if guild:
                        for ch in guild.text_channels:
                            result["channels"].append({
                                "id": str(ch.id),
                                "name": ch.name,
                                "category": ch.category.name if ch.category else "",
                            })
            finally:
                await client.close()

        try:
            await client.start(token)
        except discord.LoginFailure:
            pass
        except Exception:
            pass
        finally:
            # Give aiohttp time to close connectors cleanly
            try:
                await asyncio.sleep(0.25)
                if hasattr(client, 'http') and client.http:
                    if hasattr(client.http, 'connector') and client.http.connector and not client.http.connector.closed:
                        await client.http.connector.close()
                    # Also close the websocket connector if present
                    if hasattr(client, 'ws') and client.ws and hasattr(client.ws, 'socket'):
                        await client.ws.socket.close()
            except Exception:
                pass
            # Final sleep to let the event loop clean up
            await asyncio.sleep(0.1)

    # Suppress ResourceWarning from aiohttp's garbage-collected connectors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        # Also suppress at the warnings module level for the gc finalizer
        import gc
        gc.collect()  # Clean up before
        asyncio.run(_discover())
        gc.collect()  # Clean up after — but warnings are suppressed

    return result


# ─── API Server Commands ─────────────────────────────────────────

def cmd_api_start(host: str = "0.0.0.0", port: int = 7700):
    """Start the Alfred API server."""
    from rich.console import Console

    console = Console()

    try:
        import uvicorn
    except ImportError:
        console.print("\n  [red]FastAPI/uvicorn not installed.[/]")
        console.print("  Run: pip install fastapi uvicorn\n")
        return

    from core.config import CONFIG_FILE
    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    console.print(f"\n  [bold cyan]Alfred API Server[/]")
    console.print(f"  Listening on: [bold]http://{host}:{port}[/]")
    console.print(f"  Docs:         [bold]http://{host}:{port}/docs[/]")
    console.print(f"  Press Ctrl+C to stop.\n")

    from core.api import create_app
    from core.logging import setup_logging

    setup_logging(to_console=True, to_file=True)

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_discord_setup():
    """Interactive wizard to configure the Discord bot."""
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.rule import Rule
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config, _save_config

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    discord_cfg = cfg.get("discord", {})

    console.print()
    console.print(Rule("[bold]Discord Bot Setup", style="cyan"))
    console.print()

    # ── Step 1: Bot token ──
    existing_token = discord_cfg.get("bot_token", "")
    if existing_token:
        masked = existing_token[:8] + "..." + existing_token[-4:]
        console.print(f"  Existing token: {masked}")
        reuse = Confirm.ask("  Keep existing token?", default=True)
        if reuse:
            token = existing_token
        else:
            token = Prompt.ask("  Bot token").strip()
    else:
        console.print("  [dim]Create a bot at https://discord.com/developers/applications[/]")
        console.print("  [dim]Bot → Token → Copy[/]")
        token = Prompt.ask("  Bot token").strip()

    if not token:
        console.print("\n  [red]No token provided. Aborting.[/]\n")
        return

    # ── Step 2: Discover guilds ──
    console.print("\n  [dim]Connecting to Discord...[/]")
    discovery = _discord_discover(token)
    guilds = discovery["guilds"]

    if not guilds:
        console.print("  [red]Could not connect. Check your bot token.[/]\n")
        return

    console.print(f"  [green]Connected! Found {len(guilds)} guild(s).[/]\n")

    # Select guild
    if len(guilds) == 1:
        guild = guilds[0]
        console.print(f"  Guild: [bold]{guild['name']}[/] ({guild['id']})")
    else:
        for i, g in enumerate(guilds, 1):
            console.print(f"    [cyan]{i}.[/] {g['name']} ({g['id']})")
        choice = Prompt.ask("  Select guild", default="1")
        idx = int(choice) - 1 if choice.isdigit() else 0
        idx = max(0, min(idx, len(guilds) - 1))
        guild = guilds[idx]

    guild_id = guild["id"]

    # ── Step 3: Discover channels ──
    console.print(f"\n  [dim]Fetching channels from {guild['name']}...[/]")
    discovery = _discord_discover(token, guild_id)
    channels = discovery["channels"]

    if not channels:
        console.print("  [red]No text channels found.[/]\n")
        return

    console.print(f"  Found {len(channels)} text channel(s).\n")

    # ── Step 4: Get available agents ──
    agents = cfg.get("agents", {})
    agent_names = list(agents.keys())

    if not agent_names:
        console.print("  [yellow]No agents configured yet.[/]")
        console.print("  Create agents first: [bold]alfred agent create <name>[/]\n")
        return

    console.print("  Available agents:")
    for i, aname in enumerate(agent_names, 1):
        desc = agents[aname].get("description", "")
        console.print(f"    [cyan]{i}.[/] {aname}  [dim]{desc}[/]")
    console.print()

    # ── Step 5: Map channels → agents ──
    console.print("  [bold]Channel → Agent Mapping[/]")
    console.print("  [dim]Enter agent number for each channel, or 'skip' to ignore.[/]\n")

    channel_mappings = {}
    existing_channels = discord_cfg.get("channels", {})

    for ch in channels:
        # Check if there's an existing mapping
        existing = existing_channels.get(ch["id"], {})
        existing_agent = existing.get("agent", "")
        existing_mention = existing.get("require_mention", True)

        # Show channel
        cat_str = f" [{ch['category']}]" if ch['category'] else ""
        default_hint = f" [dim](currently: {existing_agent})[/]" if existing_agent else ""
        console.print(f"  #{ch['name']}{cat_str}{default_hint}")

        # Agent selection
        if existing_agent and existing_agent in agent_names:
            default_val = str(agent_names.index(existing_agent) + 1)
        else:
            default_val = "skip"

        choice = Prompt.ask(
            "    Agent",
            default=default_val,
        )

        if choice.lower() == "skip":
            continue

        # Parse agent choice
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(agent_names):
                agent_name = agent_names[idx]
            else:
                console.print("    [yellow]Invalid number, skipping.[/]")
                continue
        elif choice in agent_names:
            agent_name = choice
        else:
            console.print("    [yellow]Unknown agent, skipping.[/]")
            continue

        # Require mention?
        mention_default = existing_mention if existing_agent else False
        require_mention = Confirm.ask(
            "    Require @mention?",
            default=mention_default,
        )

        channel_mappings[ch["id"]] = {
            "name": ch["name"],
            "agent": agent_name,
            "require_mention": require_mention,
        }

    if not channel_mappings:
        console.print("\n  [yellow]No channels mapped. Nothing saved.[/]\n")
        return

    # ── Step 6: Save config ──
    cfg["discord"] = {
        "bot_token": token,
        "guild_id": guild_id,
        "channels": channel_mappings,
    }
    _save_config(cfg)

    # Show summary
    console.print()
    table = Table(box=box.ROUNDED, title="Discord Configuration", title_style="bold green")
    table.add_column("Channel", style="bold")
    table.add_column("Agent", style="cyan")
    table.add_column("Mode")

    for ch_id, ch_cfg in channel_mappings.items():
        mode = "@mention only" if ch_cfg["require_mention"] else "all messages"
        table.add_row(f"#{ch_cfg['name']}", ch_cfg["agent"], mode)

    console.print(table)
    console.print(f"\n  [bold green]Discord bot configured![/]")
    console.print(f"  Start it: [bold]alfred discord start[/]\n")


def cmd_start(foreground: bool = False, _daemon_child: bool = False, port: int = 7700):
    """Start Alfred — launches all services (API + scheduler + Discord)."""
    import json
    import subprocess
    import threading
    from pathlib import Path
    from rich.console import Console

    console = Console()

    # Read config without importing heavy core modules (lancedb is not fork-safe)
    config_path = Path(__file__).parent / "alfred.json"
    if not config_path.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    with open(config_path) as f:
        cfg = json.load(f)

    discord_cfg = cfg.get("discord", {})
    has_discord = discord_cfg.get("bot_token") and discord_cfg.get("channels")

    # Check PID file (skip if we ARE the daemon child)
    pid_file = Path(__file__).parent / "data" / "discord.pid"
    if not _daemon_child:
        # Check PID file first
        if pid_file.exists():
            try:
                existing_pid = int(pid_file.read_text().strip())
                os.kill(existing_pid, 0)  # Check if alive
                console.print(f"\n  [yellow]Alfred is already running (PID {existing_pid}).[/]")
                console.print("  Stop it first: [bold]alfred stop[/]\n")
                return
            except (ValueError, ProcessLookupError, PermissionError):
                pid_file.unlink(missing_ok=True)

        # Also scan for orphaned daemon-child processes not in the PID file
        from core.discord import _find_daemon_pids
        orphans = _find_daemon_pids()
        if orphans:
            console.print(f"\n  [yellow]Found orphaned Alfred process(es): {orphans}[/]")
            console.print("  Cleaning up...")
            from core.discord import _kill_and_wait
            for opid in orphans:
                _kill_and_wait(opid)
            console.print("  [green]Done.[/]\n")

    if foreground or _daemon_child:
        # Direct run — import and start services
        from core.scheduler import Scheduler
        from core.agent import Agent, AgentConfig
        from core.config import _load_config as _reload_config
        from core.logging import setup_logging
        from pathlib import Path as _Path

        _discord_bot = [None]  # mutable ref so _run_agent_task can access bot lazily

        def _run_agent_task(agent_name: str, task: str) -> str:
            """Scheduler callback — creates an agent and runs a task."""
            _cfg = _reload_config()
            agent_data = _cfg.get("agents", {}).get(agent_name)
            if not agent_data:
                raise ValueError(f"Agent '{agent_name}' not found")

            agent_data = dict(agent_data)
            agent_data["name"] = agent_name

            from core.config import config as _config
            workspace = _Path(agent_data.get("workspace", f"workspaces/{agent_name}"))
            if not workspace.is_absolute():
                workspace = _config.PROJECT_ROOT / workspace
            agent_data["workspace"] = str(workspace)
            workspace.mkdir(parents=True, exist_ok=True)

            agent_config = AgentConfig.from_dict(agent_data)
            agent = Agent(agent_config)
            result = agent.run(task)

            # Post the agent's response to its mapped Discord channel
            if _discord_bot[0] and result:
                try:
                    _discord_bot[0].post_to_agent_channel(agent_name, result)
                except Exception:
                    pass  # Don't let Discord failures kill the scheduled task

            return result

        # ── 1. Scheduler (background thread) ──
        scheduler = Scheduler(agent_runner=_run_agent_task)
        scheduler.start()

        # ── 2. API server (daemon thread) ──
        api_started = False
        try:
            import uvicorn
            import socket
            from core.api import create_app

            # Wait for port to be free (handles restart overlap)
            for _attempt in range(10):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(("127.0.0.1", port)) != 0:
                        break  # Port is free
                import time as _t; _t.sleep(1)

            # Write PID file AFTER the old daemon is gone (port freed).
            # This avoids a race where alfred stop kills us while the
            # old process is still alive.
            if _daemon_child:
                pid_file.write_text(str(os.getpid()))

            app = create_app()
            app.state.scheduler = scheduler
            uvi_config = uvicorn.Config(
                app, host="0.0.0.0", port=port,
                log_level="warning" if _daemon_child else "info",
            )
            server = uvicorn.Server(uvi_config)
            api_thread = threading.Thread(target=server.run, daemon=True, name="api-server")
            api_thread.start()
            api_started = True
        except ImportError:
            # No uvicorn — still need to write PID file
            if _daemon_child:
                pid_file.write_text(str(os.getpid()))

        # ── 3. Main blocking service ──
        if has_discord:
            from core.discord import DiscordBot

            bot = DiscordBot()
            _discord_bot[0] = bot  # allow scheduler tasks to post to Discord

            if not _daemon_child:
                services = []
                if api_started:
                    services.append(f"API on [bold]http://localhost:{port}[/]")
                channels = discord_cfg.get("channels", {})
                channel_list = ", ".join(f"#{c.get('name', '?')}" for c in channels.values())
                services.append(f"Discord ({channel_list})")
                services.append("Scheduler")
                console.print()
                for s in services:
                    console.print(f"  [green]✓[/] {s}")
                console.print(f"\n  Press Ctrl+C to stop.\n")

            try:
                bot.run(foreground=not _daemon_child)
            finally:
                scheduler.stop()
        else:
            # No Discord — API + Scheduler only. Block on signal.
            import signal as _signal

            if not _daemon_child:
                services = []
                if api_started:
                    services.append(f"API on [bold]http://localhost:{port}[/]")
                services.append("Scheduler")
                console.print()
                for s in services:
                    console.print(f"  [green]✓[/] {s}")
                console.print(f"\n  [dim]Discord not configured — run: alfred discord setup[/]")
                console.print(f"  Press Ctrl+C to stop.\n")

            # Write PID file so 'alfred stop' can find us
            # (daemon-child already wrote it after port wait; foreground needs it here)
            _pid_file = pid_file  # reuse the pid_file from outer scope
            if not _daemon_child:
                pid_file.parent.mkdir(parents=True, exist_ok=True)
                pid_file.write_text(str(os.getpid()))

            def _shutdown(signum, frame):
                raise SystemExit(0)

            _signal.signal(_signal.SIGTERM, _shutdown)
            _signal.signal(_signal.SIGINT, _shutdown)

            try:
                while True:
                    _signal.pause()
            except (SystemExit, KeyboardInterrupt):
                if not _daemon_child:
                    console.print("\n  Shutting down...")
            finally:
                # Only remove PID file if it still belongs to us (reload safety)
                try:
                    if _pid_file.exists() and int(_pid_file.read_text().strip()) == os.getpid():
                        _pid_file.unlink(missing_ok=True)
                except (ValueError, OSError):
                    _pid_file.unlink(missing_ok=True)
                scheduler.stop()
    else:
        # Daemon mode — spawn a clean subprocess (lancedb is not fork-safe)
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        log_file = data_dir / "alfred.log"

        # Find the alfred launcher script
        alfred_script = Path(__file__).parent / "alfred"

        # Pass --port through to daemon child
        daemon_args = [str(alfred_script), "start", "--daemon-child"]
        if port != 7700:
            daemon_args += ["--port", str(port)]

        lf = open(log_file, "a")
        proc = subprocess.Popen(
            daemon_args,
            stdout=lf,
            stderr=lf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # Fully detach from terminal
        )
        lf.close()

        # Write PID file
        pid_file.write_text(str(proc.pid))

        console.print(f"\n  Alfred started (PID {proc.pid})")
        console.print(f"  API:    [bold]http://localhost:{port}[/]")
        console.print(f"  Logs:   alfred logs")
        console.print(f"  Status: alfred status")
        console.print(f"  Stop:   alfred stop\n")


def cmd_stop():
    """Stop Alfred — shuts down all running services."""
    from rich.console import Console
    from core.discord import stop_bot, is_bot_running

    console = Console()

    pid = is_bot_running()
    if pid is None:
        console.print("\n  [dim]Alfred is not running.[/]\n")
        return

    console.print(f"\n  Stopping Alfred (PID {pid})...")
    if stop_bot():
        console.print("  [green]Stopped.[/]\n")
    else:
        console.print(f"  [red]Failed to stop. Try: kill {pid}[/]\n")


def cmd_logs():
    """Tail the Alfred log file."""
    from core.logging import LOG_FILE
    import subprocess

    if not LOG_FILE.exists():
        print("\n  No log file yet. Start Alfred first: alfred start\n")
        return

    print(f"  Tailing {LOG_FILE} (Ctrl+C to stop)\n")
    try:
        subprocess.run(["tail", "-f", str(LOG_FILE)])
    except KeyboardInterrupt:
        print()


def cmd_discord_status():
    """Show current Discord configuration."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    discord_cfg = cfg.get("discord", {})

    if not discord_cfg:
        console.print("\n  [dim]Discord not configured.[/]")
        console.print("  Run: [bold]alfred discord setup[/]\n")
        return

    console.print()

    # Token
    token = discord_cfg.get("bot_token", "")
    if token:
        masked = token[:8] + "..." + token[-4:]
    else:
        masked = "[red]not set[/]"

    # Summary
    guild_id = discord_cfg.get("guild_id", "?")
    console.print(f"  Token:    {masked}")
    console.print(f"  Guild ID: {guild_id}")

    # Channel table
    channels = discord_cfg.get("channels", {})
    if channels:
        console.print()
        table = Table(box=box.ROUNDED, title="Channel → Agent", title_style="bold cyan")
        table.add_column("Channel", style="bold")
        table.add_column("Channel ID", style="dim")
        table.add_column("Agent", style="cyan")
        table.add_column("Mode")

        for ch_id, ch_cfg in channels.items():
            mode = "@mention only" if ch_cfg.get("require_mention", True) else "all messages"
            table.add_row(
                f"#{ch_cfg.get('name', '?')}",
                ch_id,
                ch_cfg.get("agent", "?"),
                mode,
            )

        console.print(table)
    else:
        console.print("\n  [yellow]No channels mapped.[/]")

    # Rate limit config
    rl_cfg = discord_cfg.get("rate_limit", {})
    rl_msgs = rl_cfg.get("messages_per_minute", 5)
    rl_window = rl_cfg.get("window_seconds", 60)
    rl_cooldown = rl_cfg.get("cooldown_seconds", 30)
    console.print(f"\n  Rate limit: [bold]{rl_msgs}[/] msgs / {rl_window}s window, {rl_cooldown}s cooldown")
    console.print(f"  [dim](edit discord.rate_limit in alfred.json to customize)[/]")

    console.print(f"\n  [dim]Reconfigure: alfred discord setup[/]\n")


def cmd_discord_channel_add():
    """Map a Discord channel to an agent."""
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.rule import Rule
    from core.config import CONFIG_FILE, _load_config, _save_config

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    discord_cfg = cfg.get("discord", {})
    token = discord_cfg.get("bot_token", "")
    guild_id = discord_cfg.get("guild_id", "")

    if not token or not guild_id:
        console.print("\n  [yellow]Discord not configured yet.[/]")
        console.print("  Run: [bold]alfred discord setup[/]\n")
        return

    agents = cfg.get("agents", {})
    agent_names = list(agents.keys())

    if not agent_names:
        console.print("\n  [yellow]No agents configured.[/]")
        console.print("  Create one first: [bold]alfred agent create <name>[/]\n")
        return

    console.print()
    console.print(Rule("[bold]Add Channel Mapping", style="cyan"))
    console.print()

    # Discover channels from Discord
    console.print("  [dim]Connecting to Discord...[/]")
    discovery = _discord_discover(token, guild_id)
    channels = discovery.get("channels", [])

    if not channels:
        console.print("  [red]No text channels found in the guild.[/]\n")
        return

    # Filter to unmapped channels only
    existing_ids = set(discord_cfg.get("channels", {}).keys())
    unmapped = [ch for ch in channels if ch["id"] not in existing_ids]

    if not unmapped:
        console.print("  [yellow]All channels are already mapped![/]")
        console.print("  Create a new channel in Discord, or remove an existing mapping first.")
        console.print("  Run: [bold]alfred discord channel remove[/]\n")
        return

    console.print(f"  Found {len(unmapped)} unmapped channel(s):\n")

    for i, ch in enumerate(unmapped, 1):
        cat_str = f" [dim][{ch['category']}][/]" if ch.get("category") else ""
        console.print(f"    [cyan]{i}.[/] #{ch['name']}{cat_str}")

    console.print()
    choice = Prompt.ask("  Select channel", default="1")
    idx = int(choice) - 1 if choice.isdigit() else 0
    idx = max(0, min(idx, len(unmapped) - 1))
    selected = unmapped[idx]
    console.print(f"  Selected: [bold]#{selected['name']}[/]")

    # Agent selection
    console.print()
    if len(agent_names) == 1:
        agent_name = agent_names[0]
        desc = agents[agent_name].get("description", "")
        console.print(f"  Agent: [bold]{agent_name}[/]  [dim]{desc}[/]")
        if not Confirm.ask("  Use this agent?", default=True):
            console.print("  [dim]Cancelled.[/]\n")
            return
    else:
        console.print("  Available agents:")
        for i, aname in enumerate(agent_names, 1):
            desc = agents[aname].get("description", "")
            console.print(f"    [cyan]{i}.[/] {aname}  [dim]{desc}[/]")
        console.print()
        choice = Prompt.ask("  Select agent", default="1")
        if choice.isdigit():
            aidx = int(choice) - 1
            if 0 <= aidx < len(agent_names):
                agent_name = agent_names[aidx]
            else:
                console.print("  [red]Invalid selection.[/]\n")
                return
        elif choice in agent_names:
            agent_name = choice
        else:
            console.print("  [red]Unknown agent.[/]\n")
            return

    require_mention = Confirm.ask("  Require @mention?", default=False)

    # Save
    if "channels" not in cfg["discord"]:
        cfg["discord"]["channels"] = {}

    cfg["discord"]["channels"][selected["id"]] = {
        "name": selected["name"],
        "agent": agent_name,
        "require_mention": require_mention,
    }
    _save_config(cfg)

    mode = "@mention only" if require_mention else "all messages"
    console.print(f"\n  [bold green]Channel mapped![/]")
    console.print(f"  #{selected['name']} → {agent_name} ({mode})")
    console.print(f"\n  [dim]Restart alfred for changes to take effect.[/]\n")


def cmd_discord_channel_remove():
    """Remove a Discord channel-to-agent mapping."""
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.rule import Rule
    from core.config import CONFIG_FILE, _load_config, _save_config

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    discord_cfg = cfg.get("discord", {})

    if not discord_cfg.get("bot_token"):
        console.print("\n  [yellow]Discord not configured yet.[/]")
        console.print("  Run: [bold]alfred discord setup[/]\n")
        return

    channels = discord_cfg.get("channels", {})
    if not channels:
        console.print("\n  [yellow]No channel mappings to remove.[/]")
        console.print("  Run: [bold]alfred discord channel add[/]\n")
        return

    console.print()
    console.print(Rule("[bold]Remove Channel Mapping", style="cyan"))
    console.print()

    channel_list = list(channels.items())
    for i, (ch_id, ch_cfg) in enumerate(channel_list, 1):
        mode = "@mention only" if ch_cfg.get("require_mention", True) else "all messages"
        console.print(f"    [cyan]{i}.[/] #{ch_cfg.get('name', '?')} → {ch_cfg.get('agent', '?')} ({mode})")

    console.print()
    choice = Prompt.ask("  Select channel to remove", default="1")
    idx = int(choice) - 1 if choice.isdigit() else 0
    idx = max(0, min(idx, len(channel_list) - 1))
    ch_id, ch_cfg = channel_list[idx]

    console.print(f"\n  Remove [bold]#{ch_cfg.get('name', '?')}[/] → {ch_cfg.get('agent', '?')}?")
    if not Confirm.ask("  Confirm?", default=True):
        console.print("  [dim]Cancelled.[/]\n")
        return

    del cfg["discord"]["channels"][ch_id]
    _save_config(cfg)

    console.print(f"\n  [green]Channel mapping removed.[/]")
    console.print(f"\n  [dim]Restart alfred for changes to take effect.[/]\n")


def cmd_discord_channel_list():
    """Show current Discord channel-to-agent mappings."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config

    console = Console()

    if not CONFIG_FILE.exists():
        console.print("\n  [yellow]Run 'alfred setup' first.[/]\n")
        return

    cfg = _load_config()
    discord_cfg = cfg.get("discord", {})

    if not discord_cfg.get("bot_token"):
        console.print("\n  [yellow]Discord not configured yet.[/]")
        console.print("  Run: [bold]alfred discord setup[/]\n")
        return

    channels = discord_cfg.get("channels", {})
    if not channels:
        console.print("\n  [dim]No channel mappings configured.[/]")
        console.print("  Run: [bold]alfred discord channel add[/]\n")
        return

    console.print()
    table = Table(box=box.ROUNDED, title="Channel → Agent Mappings", title_style="bold cyan")
    table.add_column("Channel", style="bold")
    table.add_column("Channel ID", style="dim")
    table.add_column("Agent", style="cyan")
    table.add_column("Mode")

    for ch_id, ch_cfg in channels.items():
        mode = "@mention only" if ch_cfg.get("require_mention", True) else "all messages"
        table.add_row(
            f"#{ch_cfg.get('name', '?')}",
            ch_id,
            ch_cfg.get("agent", "?"),
            mode,
        )

    console.print(table)
    console.print()


# ─── Main Router ─────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    # Handle 'agent' subcommands
    if command == "agent":
        if len(sys.argv) < 3:
            print("Usage: alfred agent <create|list|info|chat|pause|resume|delete|schedule> [name]")
            return

        subcmd = sys.argv[2].lower()

        if subcmd == "create":
            if len(sys.argv) < 4:
                print("Usage: alfred agent create <name>")
                return
            cmd_agent_create(sys.argv[3])

        elif subcmd == "list":
            cmd_agent_list()

        elif subcmd == "info":
            if len(sys.argv) < 4:
                print("Usage: alfred agent info <name>")
                return
            cmd_agent_info(sys.argv[3])

        elif subcmd == "chat":
            if len(sys.argv) < 4:
                print("Usage: alfred agent chat <name> [--session NAME]")
                return
            # Parse optional --session flag
            chat_session_id = None
            chat_args = sys.argv[4:]
            for i, arg in enumerate(chat_args):
                if arg == "--session" and i + 1 < len(chat_args):
                    chat_session_id = chat_args[i + 1]
            cmd_agent_chat(sys.argv[3], session_id=chat_session_id)

        elif subcmd == "pause":
            if len(sys.argv) < 4:
                print("Usage: alfred agent pause <name>")
                return
            cmd_agent_pause(sys.argv[3])

        elif subcmd == "resume":
            if len(sys.argv) < 4:
                print("Usage: alfred agent resume <name>")
                return
            cmd_agent_resume(sys.argv[3])

        elif subcmd == "delete":
            if len(sys.argv) < 4:
                print("Usage: alfred agent delete <name>")
                return
            cmd_agent_delete(sys.argv[3])

        elif subcmd == "schedule":
            if len(sys.argv) < 4:
                print("Usage: alfred agent schedule <add|list|remove|enable|disable|run|history|retry> [name] [id]")
                return

            sched_cmd = sys.argv[3].lower()

            if sched_cmd == "add":
                if len(sys.argv) < 5:
                    print("Usage: alfred agent schedule add <agent_name>")
                    return
                cmd_agent_schedule_add(sys.argv[4])

            elif sched_cmd == "list":
                agent_name = sys.argv[4] if len(sys.argv) > 4 else None
                cmd_agent_schedule_list(agent_name)

            elif sched_cmd == "remove":
                if len(sys.argv) < 6:
                    print("Usage: alfred agent schedule remove <agent_name> <schedule_id>")
                    return
                cmd_agent_schedule_remove(sys.argv[4], sys.argv[5])

            elif sched_cmd == "enable":
                if len(sys.argv) < 6:
                    print("Usage: alfred agent schedule enable <agent_name> <schedule_id>")
                    return
                cmd_agent_schedule_enable(sys.argv[4], sys.argv[5])

            elif sched_cmd == "disable":
                if len(sys.argv) < 6:
                    print("Usage: alfred agent schedule disable <agent_name> <schedule_id>")
                    return
                cmd_agent_schedule_disable(sys.argv[4], sys.argv[5])

            elif sched_cmd == "run":
                if len(sys.argv) < 6:
                    print("Usage: alfred agent schedule run <agent_name> <schedule_id>")
                    return
                cmd_agent_schedule_run(sys.argv[4], sys.argv[5])

            elif sched_cmd == "history":
                if len(sys.argv) < 6:
                    print("Usage: alfred agent schedule history <agent_name> <schedule_id>")
                    return
                cmd_agent_schedule_history(sys.argv[4], sys.argv[5])

            elif sched_cmd == "retry":
                if len(sys.argv) < 6:
                    print("Usage: alfred agent schedule retry <agent_name> <schedule_id>")
                    return
                cmd_agent_schedule_retry(sys.argv[4], sys.argv[5])

            else:
                print(f"Unknown schedule command: {sched_cmd}")
                print("Available: add, list, remove, enable, disable, run, history, retry")

        else:
            print(f"Unknown agent command: {subcmd}")
            print("Available: create, list, info, chat, pause, resume, delete, schedule")
        return

    # Handle 'provider' subcommands
    if command == "provider":
        if len(sys.argv) < 3:
            print("Usage: alfred provider add <anthropic|xai|openai|ollama|brave>")
            return

        subcmd = sys.argv[2].lower()
        if subcmd == "add":
            if len(sys.argv) < 4:
                print("Usage: alfred provider add <anthropic|xai|openai|ollama|brave>")
                return
            cmd_provider_add(sys.argv[3])
        else:
            print(f"Unknown provider command: {subcmd}")
            print("Available: add")
        return

    # Handle 'service' subcommands
    if command == "service":
        if len(sys.argv) < 3:
            print("Usage: alfred service <add|list|remove> [name]")
            return

        subcmd = sys.argv[2].lower()

        if subcmd == "list":
            cmd_service_list()
        elif subcmd == "add":
            if len(sys.argv) < 4:
                print("Usage: alfred service add <name>")
                return
            cmd_service_add(sys.argv[3])
        elif subcmd == "remove":
            if len(sys.argv) < 4:
                print("Usage: alfred service remove <name>")
                return
            cmd_service_remove(sys.argv[3])
        else:
            print(f"Unknown service command: {subcmd}")
            print("Available: add, list, remove")
        return

    # Handle 'tools' subcommands
    if command == "tools":
        if len(sys.argv) < 3:
            print("Usage: alfred tools list [agent_name]")
            return

        subcmd = sys.argv[2].lower()
        if subcmd == "list":
            agent_name = sys.argv[3] if len(sys.argv) > 3 else None
            cmd_tools_list(agent_name)
        else:
            print(f"Unknown tools command: {subcmd}")
            print("Available: list")
        return

    # Handle 'models' subcommands
    if command == "models":
        if len(sys.argv) < 3:
            print("Usage: alfred models <update|list> [provider]")
            return

        subcmd = sys.argv[2].lower()
        provider = sys.argv[3] if len(sys.argv) > 3 else None

        if subcmd == "update":
            cmd_models_update(provider)
        elif subcmd == "list":
            cmd_models_list(provider)
        else:
            print(f"Unknown models command: {subcmd}")
            print("Available: update, list")
        return

    # Handle 'start' command
    if command == "start":
        foreground = "--fg" in sys.argv[2:] or "--foreground" in sys.argv[2:]
        daemon_child = "--daemon-child" in sys.argv[2:]
        port = 7700
        args = sys.argv[2:]
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                try:
                    port = int(args[i + 1])
                except ValueError:
                    pass
        cmd_start(foreground=foreground, _daemon_child=daemon_child, port=port)
        return

    # Handle 'stop' command
    if command == "stop":
        cmd_stop()
        return

    # Handle 'logs' command
    if command == "logs":
        cmd_logs()
        return

    # Handle 'api' subcommands
    if command == "api":
        if len(sys.argv) < 3:
            print("Usage: alfred api start [--port PORT] [--host HOST]")
            return

        subcmd = sys.argv[2].lower()

        if subcmd == "start":
            # Parse optional --port and --host flags
            host = "0.0.0.0"
            port = 7700
            args = sys.argv[3:]
            for i, arg in enumerate(args):
                if arg == "--port" and i + 1 < len(args):
                    port = int(args[i + 1])
                elif arg == "--host" and i + 1 < len(args):
                    host = args[i + 1]
            cmd_api_start(host=host, port=port)
        else:
            print(f"Unknown api command: {subcmd}")
            print("Available: start")
        return

    # Handle 'discord' subcommands
    if command == "discord":
        if len(sys.argv) < 3:
            print("Usage: alfred discord <setup|status|channel> ...")
            return

        subcmd = sys.argv[2].lower()

        if subcmd == "setup":
            cmd_discord_setup()
        elif subcmd == "status":
            cmd_discord_status()
        elif subcmd == "channel":
            if len(sys.argv) < 4:
                print("Usage: alfred discord channel <add|remove|list>")
                return
            channel_cmd = sys.argv[3].lower()
            if channel_cmd == "add":
                cmd_discord_channel_add()
            elif channel_cmd == "remove":
                cmd_discord_channel_remove()
            elif channel_cmd == "list":
                cmd_discord_channel_list()
            else:
                print(f"Unknown discord channel command: {channel_cmd}")
                print("Available: add, remove, list")
        else:
            print(f"Unknown discord command: {subcmd}")
            print("Available: setup, status, channel")
        return

    # Handle 'session' subcommands
    if command == "session":
        if len(sys.argv) < 3:
            print("Usage: alfred session <list|view|export|delete> [agent] [session_id]")
            return

        subcmd = sys.argv[2].lower()

        if subcmd == "list":
            agent_name = sys.argv[3] if len(sys.argv) > 3 else None
            cmd_session_list(agent_name)

        elif subcmd == "view":
            if len(sys.argv) < 5:
                print("Usage: alfred session view <agent> <session_id> [--last N]")
                return
            # Parse optional --last flag
            last_n = None
            view_args = sys.argv[5:]
            for i, arg in enumerate(view_args):
                if arg == "--last" and i + 1 < len(view_args):
                    try:
                        last_n = int(view_args[i + 1])
                    except ValueError:
                        pass
            cmd_session_view(sys.argv[3], sys.argv[4], last_n=last_n)

        elif subcmd == "export":
            if len(sys.argv) < 5:
                print("Usage: alfred session export <agent> <session_id> [--output FILE] [--format text|markdown]")
                return
            output_path = None
            fmt = "markdown"
            export_args = sys.argv[5:]
            for i, arg in enumerate(export_args):
                if arg == "--output" and i + 1 < len(export_args):
                    output_path = export_args[i + 1]
                elif arg == "--format" and i + 1 < len(export_args):
                    fmt = export_args[i + 1]
            cmd_session_export(sys.argv[3], sys.argv[4], output_path=output_path, format=fmt)

        elif subcmd == "delete":
            if len(sys.argv) < 5:
                print("Usage: alfred session delete <agent> <session_id>")
                return
            cmd_session_delete(sys.argv[3], sys.argv[4])

        else:
            print(f"Unknown session command: {subcmd}")
            print("Available: list, view, export, delete")
        return

    commands = {
        "setup": cmd_setup,
        "status": cmd_status,
        "demo": cmd_demo,
    }

    if command in commands:
        commands[command]()
    elif command in ("help", "--help", "-h"):
        print_usage()
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
