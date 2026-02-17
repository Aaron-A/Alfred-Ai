"""
Alfred AI - Setup Wizard (TUI)
Interactive setup that detects available providers,
collects API keys, and generates alfred.json.

Run: python -m alfred setup
"""

import os
import sys
import json
import time

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.rule import Rule
from rich import box

from core.config import CONFIG_FILE, Config, _save_config
from core.llm import detect_ollama, LLMClient


console = Console()


# ─── Provider Definitions ────────────────────────────────────────

PROVIDERS = {
    "anthropic": {
        "name": "Anthropic (Claude)",
        "description": "Claude Sonnet, Opus, Haiku — best reasoning and code",
        "env_var": "ANTHROPIC_API_KEY",
        "key_prefix": "sk-ant-",
        "key_hint": "Starts with sk-ant-...",
        "models": [
            ("claude-sonnet-4-6", "Sonnet 4.6 — latest, fast + smart (recommended)"),
            ("claude-opus-4-6", "Opus 4.6 — latest, strongest reasoning"),
            ("claude-haiku-4-5-20251001", "Haiku 4.5 — fast + cheap"),
            ("claude-sonnet-4-5-20250929", "Sonnet 4.5 — previous gen"),
            ("claude-opus-4-5-20251101", "Opus 4.5 — previous gen"),
            ("claude-opus-4-20250514", "Opus 4 — legacy"),
            ("claude-haiku-3-5-20241022", "Haiku 3.5 — legacy, cheapest"),
        ],
        "requires_key": True,
    },
    "xai": {
        "name": "xAI (Grok)",
        "description": "Grok 4.1 — fast reasoning, web search, agentic",
        "env_var": "XAI_API_KEY",
        "key_prefix": "xai-",
        "key_hint": "Starts with xai-...",
        "models": [
            ("grok-4-1-fast-reasoning", "Grok 4.1 Fast Reasoning — latest, best for agents (recommended)"),
            ("grok-4-1-fast-non-reasoning", "Grok 4.1 Fast — instant response, no CoT"),
            ("grok-code-fast-1", "Grok Code Fast — optimized for coding"),
            ("grok-4", "Grok 4 — flagship reasoning"),
            ("grok-4-fast-reasoning", "Grok 4 Fast Reasoning — balanced"),
            ("grok-4-fast-non-reasoning", "Grok 4 Fast — no CoT"),
            ("grok-3", "Grok 3 — legacy"),
            ("grok-3-mini", "Grok 3 Mini — legacy, cheapest"),
        ],
        "requires_key": True,
    },
    "openai": {
        "name": "OpenAI (GPT)",
        "description": "GPT-5.2, Codex, o-series — widely used",
        "env_var": "OPENAI_API_KEY",
        "key_prefix": "sk-",
        "key_hint": "Starts with sk-...",
        "models": [
            ("gpt-5.2", "GPT-5.2 — flagship reasoning, 400K context (recommended)"),
            ("gpt-5.2-codex", "GPT-5.2 Codex — optimized for coding"),
            ("gpt-5.1", "GPT-5.1 — previous flagship"),
            ("gpt-5.1-codex", "GPT-5.1 Codex — previous gen coding"),
            ("gpt-4.1-2025-04-14", "GPT-4.1 — 1M context window"),
            ("gpt-4.1-mini-2025-04-14", "GPT-4.1 Mini — 1M context, cheaper"),
            ("o3-2025-04-16", "o3 — dedicated reasoning model"),
            ("o4-mini-2025-04-16", "o4 Mini — fast reasoning"),
        ],
        "requires_key": True,
    },
    "ollama": {
        "name": "Ollama (Local)",
        "description": "Run models locally — Llama, DeepSeek, Qwen, Mistral",
        "env_var": "",
        "key_prefix": "",
        "key_hint": "No API key needed",
        "models": [],  # Populated dynamically from Ollama
        "requires_key": False,
    },
}


def print_banner():
    """Print the Alfred AI banner."""
    banner = Text()
    banner.append("  _   _  ___ ___  ___ ___\n", style="bold cyan")
    banner.append(" /_\\ | ||  _| _ \\| __|   \\\n", style="bold cyan")
    banner.append("/ _ \\| || _||   /| _|| |) |\n", style="bold cyan")
    banner.append("/_/ \\_|_||_| |_|_\\|___|___/\n", style="bold cyan")
    banner.append("\n")
    banner.append("  Memory-First Agent Framework", style="dim")

    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))
    console.print()


def detect_environment() -> dict:
    """Detect available providers and existing config."""
    console.print(Rule("[bold]Detecting Environment", style="cyan"))
    console.print()

    detected = {
        "ollama_running": False,
        "ollama_models": [],
        "env_keys": {},
        "existing_config": None,
    }

    # Check for existing config
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            detected["existing_config"] = json.load(f)
        console.print("  [yellow]![/] Existing alfred.json found")

    # Check environment variables for API keys
    for provider_id, prov in PROVIDERS.items():
        if prov["env_var"]:
            key = os.getenv(prov["env_var"], "")
            if key:
                detected["env_keys"][provider_id] = key
                masked = key[:8] + "..." + key[-4:]
                console.print(f"  [green]\u2713[/] {prov['name']} key found in env: {masked}")

    # Check Ollama
    console.print("  [dim]Checking for Ollama...[/]", end="")
    ollama_running, ollama_models = detect_ollama()
    detected["ollama_running"] = ollama_running
    detected["ollama_models"] = ollama_models

    if ollama_running:
        console.print(f"\r  [green]\u2713[/] Ollama running with {len(ollama_models)} model(s)")
        if ollama_models:
            for m in ollama_models[:8]:
                console.print(f"    [dim]\u2022 {m}[/]")
            if len(ollama_models) > 8:
                console.print(f"    [dim]  ...and {len(ollama_models) - 8} more[/]")
    else:
        console.print("\r  [dim]\u2022[/] Ollama not detected (optional)")

    console.print()
    return detected


def select_providers(detected: dict) -> list[str]:
    """Let user choose which providers to configure."""
    console.print(Rule("[bold]LLM Providers", style="cyan"))
    console.print()
    console.print("  Select which LLM providers you want to use.")
    console.print("  You can configure multiple and switch between them.\n")

    # Build provider table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Provider", style="bold")
    table.add_column("Description")
    table.add_column("Status")

    provider_list = list(PROVIDERS.keys())
    for i, pid in enumerate(provider_list, 1):
        prov = PROVIDERS[pid]
        # Determine status
        if pid == "ollama":
            if detected["ollama_running"]:
                status = f"[green]\u2713 Running ({len(detected['ollama_models'])} models)[/]"
            else:
                status = "[dim]Not detected[/]"
        elif pid in detected["env_keys"]:
            status = "[green]\u2713 Key in env[/]"
        else:
            status = "[dim]Not configured[/]"

        table.add_row(str(i), prov["name"], prov["description"], status)

    console.print(table)
    console.print()

    # Ask which providers to configure
    choices = Prompt.ask(
        "  [bold]Which providers?[/] (comma-separated numbers, e.g. 1,2)",
        default="1",
    )

    selected = []
    for c in choices.split(","):
        c = c.strip()
        if c.isdigit() and 1 <= int(c) <= len(provider_list):
            selected.append(provider_list[int(c) - 1])

    if not selected:
        console.print("  [yellow]No providers selected. Using Anthropic as default.[/]")
        selected = ["anthropic"]

    console.print()
    provider_names = ", ".join(PROVIDERS[p]["name"] for p in selected)
    console.print(f"  [green]\u2713[/] Selected: {provider_names}")
    console.print()

    return selected


def configure_provider(provider_id: str, detected: dict) -> dict:
    """Configure a single provider — get API key and model selection."""
    prov = PROVIDERS[provider_id]
    result = {"enabled": True}

    console.print(Rule(f"[bold]{prov['name']}", style="cyan"))
    console.print()

    # Handle API key
    if prov["requires_key"]:
        # Check if we already have a key from env
        existing_key = detected["env_keys"].get(provider_id, "")
        if existing_key:
            masked = existing_key[:8] + "..." + existing_key[-4:]
            use_existing = Confirm.ask(
                f"  Use existing key from env? ({masked})",
                default=True,
            )
            if use_existing:
                result["api_key"] = existing_key
            else:
                key = Prompt.ask(f"  Enter {prov['name']} API key ({prov['key_hint']})")
                result["api_key"] = key.strip()
        else:
            key = Prompt.ask(f"  Enter {prov['name']} API key ({prov['key_hint']})")
            result["api_key"] = key.strip()

        if not result.get("api_key"):
            console.print(f"  [yellow]No key provided. {prov['name']} will be disabled.[/]")
            result["enabled"] = False
            console.print()
            return result

    # Handle model selection
    if provider_id == "ollama":
        # Use detected models
        models = detected.get("ollama_models", [])
        if not models:
            console.print("  [yellow]No Ollama models found. Pull one with: ollama pull llama3.1[/]")
            result["model"] = "llama3.1"
        else:
            console.print("  Available models:")
            for i, m in enumerate(models, 1):
                console.print(f"    [cyan]{i}.[/] {m}")
            choice = Prompt.ask("  Select model number", default="1")
            idx = int(choice) - 1 if choice.isdigit() else 0
            idx = max(0, min(idx, len(models) - 1))
            result["model"] = models[idx]
    else:
        # Use predefined model list
        console.print("  Available models:")
        for i, (model_id, desc) in enumerate(prov["models"], 1):
            console.print(f"    [cyan]{i}.[/] {desc}")
        choice = Prompt.ask("  Select model number", default="1")
        idx = int(choice) - 1 if choice.isdigit() else 0
        idx = max(0, min(idx, len(prov["models"]) - 1))
        result["model"] = prov["models"][idx][0]

    console.print(f"\n  [green]\u2713[/] Model: [bold]{result['model']}[/]")

    # Test connection
    if Confirm.ask("  Test connection now?", default=True):
        console.print("  [dim]Testing...[/]", end="")
        client = LLMClient(
            provider=provider_id,
            api_key=result.get("api_key", ""),
            model=result["model"],
        )
        success, msg = client.test_connection()
        if success:
            console.print(f"\r  [green]\u2713 {msg}[/]")
        else:
            console.print(f"\r  [red]\u2717 Connection failed: {msg}[/]")
            if not Confirm.ask("  Continue anyway?", default=True):
                result["enabled"] = False

    console.print()
    return result


def select_default_provider(configured: dict) -> str:
    """Choose which provider is the default."""
    enabled = [pid for pid, cfg in configured.items() if cfg.get("enabled", True)]

    if len(enabled) == 1:
        return enabled[0]

    console.print(Rule("[bold]Default Provider", style="cyan"))
    console.print()
    console.print("  Which provider should be the default for new agents?\n")

    for i, pid in enumerate(enabled, 1):
        prov = PROVIDERS[pid]
        model = configured[pid].get("model", "?")
        console.print(f"    [cyan]{i}.[/] {prov['name']} ({model})")

    choice = Prompt.ask("\n  Select default", default="1")
    idx = int(choice) - 1 if choice.isdigit() else 0
    idx = max(0, min(idx, len(enabled) - 1))

    console.print(f"\n  [green]\u2713[/] Default: [bold]{PROVIDERS[enabled[idx]]['name']}[/]")
    console.print()

    return enabled[idx]


def configure_embeddings() -> dict:
    """Configure the embedding model."""
    console.print(Rule("[bold]Embedding Model", style="cyan"))
    console.print()
    console.print("  Embeddings power Alfred's memory search.")
    console.print("  The local model runs on your machine — no API calls, fully private.\n")

    options = [
        ("nomic-ai/nomic-embed-text-v1.5", "Nomic Embed v1.5 — local, ~270MB (recommended)"),
        ("all-MiniLM-L6-v2", "MiniLM L6 — local, ~90MB, faster but less accurate"),
    ]

    for i, (model_id, desc) in enumerate(options, 1):
        console.print(f"    [cyan]{i}.[/] {desc}")

    choice = Prompt.ask("\n  Select embedding model", default="1")
    idx = int(choice) - 1 if choice.isdigit() else 0
    idx = max(0, min(idx, len(options) - 1))

    model_id, desc = options[idx]
    dim = 768 if "nomic" in model_id else 384

    console.print(f"\n  [green]\u2713[/] Embedding model: [bold]{model_id}[/]")
    console.print()

    return {"model": model_id, "dimension": dim}


def write_config(
    providers_config: dict,
    default_provider: str,
    embeddings_config: dict,
):
    """Write the final alfred.json config."""
    alfred_config = {
        "version": "0.1.0",
        "llm": {
            "provider": default_provider,
            "model": providers_config[default_provider].get("model", ""),
        },
        "providers": {},
        "embeddings": embeddings_config,
        "memory": {
            "db_path": "data/lancedb",
            "top_k": 10,
            "hybrid_search": {
                "enabled": True,
                "vector_weight": 0.7,
                "text_weight": 0.3,
            },
        },
    }

    # Add provider configs (strip sensitive keys from display, keep in file)
    for pid, cfg in providers_config.items():
        if cfg.get("enabled", True):
            provider_entry = {"model": cfg.get("model", "")}
            if "api_key" in cfg:
                provider_entry["api_key"] = cfg["api_key"]
            alfred_config["providers"][pid] = provider_entry

    _save_config(alfred_config)
    return alfred_config


def print_summary(config_data: dict):
    """Print a nice summary of what was configured."""
    console.print(Rule("[bold]Setup Complete", style="green"))
    console.print()

    table = Table(box=box.ROUNDED, title="Alfred Configuration", title_style="bold green")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    # Default LLM
    provider = config_data["llm"]["provider"]
    model = config_data["llm"]["model"]
    table.add_row("Default LLM", f"{PROVIDERS[provider]['name']}")
    table.add_row("Default Model", model)

    # All providers
    for pid, pcfg in config_data.get("providers", {}).items():
        has_key = "api_key" in pcfg
        status = f"{pcfg.get('model', '?')}"
        if pid == "ollama":
            status += " (local)"
        elif has_key:
            status += " [green](key set)[/]"
        table.add_row(f"  {PROVIDERS[pid]['name']}", status)

    # Embeddings
    emb = config_data.get("embeddings", {})
    table.add_row("Embedding Model", emb.get("model", "?"))
    table.add_row("Embedding Dim", str(emb.get("dimension", "?")))

    # Memory
    table.add_row("Vector Store", "LanceDB (data/lancedb/)")
    table.add_row("Config File", str(CONFIG_FILE))

    console.print(table)
    console.print()

    console.print("  [bold green]Alfred is ready.[/]\n")
    console.print("  [dim]Next steps:[/]")
    console.print("    alfred agent chat alfred     [dim]# Chat with your default agent[/]")
    console.print("    alfred agent create <name>   [dim]# Create a specialist agent[/]")
    console.print("    alfred discord setup          [dim]# Connect to Discord[/]")
    console.print()


def create_default_agent(config_data: dict):
    """Create the default 'alfred' agent — always present from first setup."""
    from core.config import _load_config, _save_config, config
    from core.workspace import create_workspace
    from core.tools import ToolRegistry, register_builtin_tools
    from core.tool_discovery import discover_shared_tools

    console.print(Rule("[bold]Default Agent", style="cyan"))
    console.print()
    console.print("  Creating your default agent: [bold cyan]alfred[/]")
    console.print("  [dim]He'll be available in all channels unless you assign a specialist.[/]")
    console.print()

    # Build registry for TOOLS.md generation
    registry = ToolRegistry()
    register_builtin_tools(registry)
    try:
        discover_shared_tools(registry)
    except Exception:
        pass  # Shared tools are optional

    # Create workspace
    workspace_path = str(config.PROJECT_ROOT / "workspaces" / "alfred")
    created = create_workspace(workspace_path, "alfred", registry=registry)
    if created:
        for f in created:
            console.print(f"    [dim]{f}[/]")

    # Add to config
    cfg = _load_config()
    if "agents" not in cfg:
        cfg["agents"] = {}

    cfg["agents"]["alfred"] = {
        "workspace": "workspaces/alfred",
        "description": "General-purpose assistant — your default agent",
        "status": "active",
    }
    _save_config(cfg)

    console.print(f"\n  [green]✓[/] Agent [bold]alfred[/] ready")
    console.print(f"    Chat: [bold]alfred agent chat alfred[/]")
    console.print(f"    Rename: edit alfred.json or create a new agent")
    console.print()


def run_setup():
    """Main setup wizard entry point."""
    print_banner()

    # Check for existing config
    if CONFIG_FILE.exists():
        console.print(f"  [yellow]Existing configuration found at {CONFIG_FILE}[/]")
        if not Confirm.ask("  Overwrite existing configuration?", default=False):
            console.print("  [dim]Setup cancelled.[/]")
            return

    console.print()

    # Step 1: Detect environment
    detected = detect_environment()

    # Step 2: Select providers
    selected = select_providers(detected)

    # Step 3: Configure each provider
    providers_config = {}
    for pid in selected:
        providers_config[pid] = configure_provider(pid, detected)

    # Step 4: Select default provider
    default = select_default_provider(providers_config)

    # Step 5: Configure embeddings
    embeddings = configure_embeddings()

    # Step 6: Write config
    config_data = write_config(providers_config, default, embeddings)

    # Step 7: Create default "alfred" agent
    create_default_agent(config_data)

    # Step 8: Summary
    print_summary(config_data)


if __name__ == "__main__":
    run_setup()
