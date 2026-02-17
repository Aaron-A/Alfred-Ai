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

    discord setup      - Configure Discord bot (token, guild, channel→agent mapping)
    discord status     - Show current Discord configuration

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
    start                               Start Alfred (background daemon)
    start --fg                          Start in foreground (Ctrl+C to stop)
    stop                                Stop Alfred
    status                              Show configuration and running state
    logs                                Tail the log file

    provider add <name>                 Add an LLM provider (anthropic, xai, openai, ollama)

    models update [provider]            Fetch latest models from provider APIs
    models list [provider]              Show available models (cached)

    agent create <name>                 Create a new agent with workspace
    agent list                          List all configured agents
    agent info <name>                   Show detailed agent info
    agent chat <name>                   Interactive chat with an agent
    agent pause <name>                  Pause an agent (disables schedules)
    agent resume <name>                 Resume a paused agent
    agent delete <name>                 Delete an agent and its config

    agent schedule add <name>           Add a scheduled task to an agent
    agent schedule list [name]          List scheduled tasks (all or one agent)
    agent schedule remove <name> <id>   Remove a scheduled task

    tools list [agent_name]             List all registered tools
                                        (omit name for global view, add name for agent-specific)

    discord setup                       Configure Discord bot (token, channels, agents)
    discord status                      Show Discord configuration

    api start [--port PORT]             Start the HTTP API server (default: 7700)

    demo                                Run the memory layer demo

Examples:
    alfred setup                                    # First-time setup
    alfred start                                    # Start Alfred (Discord bot, etc.)
    alfred stop                                     # Stop Alfred
    alfred status                                   # Check what's running
    alfred logs                                     # Watch the log
    alfred provider add anthropic                   # Add Claude as a provider
    alfred agent create trader                      # Create a trading agent
    alfred agent chat trader                        # Chat with it
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

    # Discord
    discord_cfg = cfg.get("discord", {})
    if discord_cfg.get("bot_token"):
        from core.discord import is_bot_running
        pid = is_bot_running()
        channels = discord_cfg.get("channels", {})
        channel_list = ", ".join(
            f"#{c.get('name', '?')}→{c.get('agent', '?')}"
            for c in channels.values()
        )
        if pid:
            status_str = f"[bold green]running[/] (PID {pid})"
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
        console.print()
        sched_table = Table(box=box.ROUNDED, title="Schedules", title_style="bold")
        sched_table.add_column("ID", style="cyan", width=10)
        sched_table.add_column("Cron")
        sched_table.add_column("Schedule")
        sched_table.add_column("Task")
        sched_table.add_column("Status")
        sched_table.add_column("Last Run")

        for s in schedules:
            sid = s.get("id", "?")
            cron = s.get("cron", "?")
            enabled = s.get("enabled", True)
            task = s.get("task", "?")
            if len(task) > 40:
                task = task[:37] + "..."
            last_run = s.get("last_run", "")
            last_result = s.get("last_result", "")

            status = "[green]active[/]" if enabled else "[yellow]paused[/]"
            if last_run:
                # Show just date+time, not full ISO
                run_display = last_run[:16].replace("T", " ")
                if last_result and "error" in last_result:
                    run_display += f" [red]({last_result[:20]})[/]"
            else:
                run_display = "[dim]never[/]"

            sched_table.add_row(sid, f"[dim]{cron}[/]", describe_cron(cron), task, status, run_display)

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


def cmd_agent_chat(name: str):
    from rich.console import Console
    from rich.panel import Panel
    from core.config import CONFIG_FILE, _load_config
    from core.agent import AgentManager

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

    # Load agent
    manager = AgentManager()
    agent = manager.get(name)

    console.print()
    session = agent.session_info
    session_status = ""
    if session["turns"] > 0:
        session_status = f" | [green]{session['turns']} turns restored[/]"
    console.print(Panel(
        f"[bold cyan]{name}[/] | {agent.llm.provider}/{agent.llm.model} | "
        f"{len(agent._get_available_tools())} tools{session_status}\n"
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
    from rich.prompt import Prompt
    from rich.rule import Rule
    from core.config import CONFIG_FILE, _load_config
    from core.scheduler import add_schedule, describe_cron, cron_matches

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

    # Task description
    console.print()
    task = Prompt.ask("  Task (what should the agent do?)")

    schedule = add_schedule(name, cron, task)

    console.print(f"\n  [green]Schedule added![/]")
    console.print(f"  ID: [cyan]{schedule.id}[/]")
    console.print(f"  When: {human}")
    console.print(f"  Task: {task}")
    console.print()


def cmd_agent_schedule_list(name: str = None):
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from core.config import CONFIG_FILE, _load_config
    from core.scheduler import describe_cron

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
        table.add_column("Cron", style="dim")
        table.add_column("Task")
        table.add_column("Status")
        table.add_column("Last Run")

        for s in schedules:
            sid = s.get("id", "?")
            cron = s.get("cron", "?")
            enabled = s.get("enabled", True)
            task = s.get("task", "?")
            if len(task) > 45:
                task = task[:42] + "..."
            last_run = s.get("last_run", "")

            status = "[green]active[/]" if enabled else "[yellow]paused[/]"
            if agent_status == "paused":
                status = "[yellow]agent paused[/]"

            run_display = last_run[:16].replace("T", " ") if last_run else "[dim]never[/]"

            table.add_row(sid, describe_cron(cron), cron, task, status, run_display)

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
        console.print(f"  Available: {', '.join(PROVIDERS.keys())}\n")
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


def cmd_demo():
    import demo
    demo.main()


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


def cmd_start(foreground: bool = False, _daemon_child: bool = False):
    """Start Alfred — launches all configured services (Discord bot, etc.)."""
    import json
    import subprocess
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

    if not has_discord:
        console.print("\n  [yellow]Nothing to start.[/]")
        console.print("  Set up Discord first: [bold]alfred discord setup[/]\n")
        return

    # Check PID file (skip if we ARE the daemon child)
    pid_file = Path(__file__).parent / "data" / "discord.pid"
    if not _daemon_child and pid_file.exists():
        try:
            existing_pid = int(pid_file.read_text().strip())
            os.kill(existing_pid, 0)  # Check if alive
            console.print(f"\n  [yellow]Alfred is already running (PID {existing_pid}).[/]")
            console.print("  Stop it first: [bold]alfred stop[/]\n")
            return
        except (ValueError, ProcessLookupError, PermissionError):
            pid_file.unlink(missing_ok=True)

    if foreground or _daemon_child:
        # Direct run — import and start services
        from core.discord import DiscordBot

        # Start the scheduler (background thread — runs cron tasks on agents)
        from core.scheduler import Scheduler
        from core.agent import Agent, AgentConfig
        from core.config import _load_config as _reload_config
        from pathlib import Path as _Path

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
            return agent.run(task)

        scheduler = Scheduler(agent_runner=_run_agent_task)
        scheduler.start()

        bot = DiscordBot()

        if not _daemon_child:
            channels = discord_cfg.get("channels", {})
            channel_list = ", ".join(f"#{c.get('name', '?')}" for c in channels.values())
            console.print(f"\n  Discord: {channel_list}\n")

        try:
            bot.run(foreground=not _daemon_child)
        finally:
            scheduler.stop()
    else:
        # Daemon mode — spawn a clean subprocess (lancedb is not fork-safe)
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        log_file = data_dir / "alfred.log"

        # Find the alfred launcher script
        alfred_script = Path(__file__).parent / "alfred"

        lf = open(log_file, "a")
        proc = subprocess.Popen(
            [str(alfred_script), "start", "--daemon-child"],
            stdout=lf,
            stderr=lf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # Fully detach from terminal
        )
        lf.close()

        # Write PID file
        pid_file.write_text(str(proc.pid))

        console.print(f"\n  Alfred started (PID {proc.pid})")
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
                print("Usage: alfred agent chat <name>")
                return
            cmd_agent_chat(sys.argv[3])

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
                print("Usage: alfred agent schedule <add|list|remove> [name] [id]")
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

            else:
                print(f"Unknown schedule command: {sched_cmd}")
                print("Available: add, list, remove")

        else:
            print(f"Unknown agent command: {subcmd}")
            print("Available: create, list, info, chat, pause, resume, delete, schedule")
        return

    # Handle 'provider' subcommands
    if command == "provider":
        if len(sys.argv) < 3:
            print("Usage: alfred provider add <anthropic|xai|openai|ollama>")
            return

        subcmd = sys.argv[2].lower()
        if subcmd == "add":
            if len(sys.argv) < 4:
                print("Usage: alfred provider add <anthropic|xai|openai|ollama>")
                return
            cmd_provider_add(sys.argv[3])
        else:
            print(f"Unknown provider command: {subcmd}")
            print("Available: add")
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
        cmd_start(foreground=foreground, _daemon_child=daemon_child)
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
            print("Usage: alfred discord <setup|status>")
            return

        subcmd = sys.argv[2].lower()

        if subcmd == "setup":
            cmd_discord_setup()
        elif subcmd == "status":
            cmd_discord_status()
        else:
            print(f"Unknown discord command: {subcmd}")
            print("Available: setup, status")
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
