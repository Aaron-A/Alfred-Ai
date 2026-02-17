# Alfred AI

A memory-first agent framework in Python. Create AI agents that remember, learn, and act — with built-in Discord integration.

## What is Alfred?

Alfred is a lightweight framework for building persistent AI agents. Each agent has:

- **Vector memory** — automatic context recall powered by LanceDB + hybrid search
- **Tools** — builtin, shared, and per-agent toolkits with auto-discovery
- **A workspace** — persistent files that define the agent's identity, knowledge, and state
- **Multi-provider LLM support** — Anthropic, xAI, and more
- **Discord integration** — map agents to Discord channels with one command

## Quick Start

```bash
# Clone and set up
git clone https://github.com/Aaron-A/Alfred-Ai.git
cd Alfred-Ai
python3 -m venv venv
source venv/bin/activate
pip install anthropic lancedb sentence-transformers discord.py rich

# Symlink the CLI (optional, for global access)
sudo ln -sf "$(pwd)/alfred" /usr/local/bin/alfred

# Run the setup wizard
alfred setup
```

The setup wizard walks you through:
1. Configuring your LLM provider and API key
2. Selecting a model
3. Initializing the embedding model for memory
4. Creating your first agent (Alfred, by default)

## CLI Reference

```
alfred setup                    Interactive setup wizard
alfred status                   Show current configuration

alfred provider add             Add an LLM provider
alfred models update            Fetch latest models from provider APIs
alfred models list              Show available models

alfred agent create             Create a new agent
alfred agent list               List all agents
alfred agent info               Show agent details
alfred agent chat               Interactive chat with an agent
alfred agent pause / resume     Pause or resume an agent
alfred agent delete             Delete an agent

alfred agent schedule add       Add a scheduled task
alfred agent schedule list      List scheduled tasks
alfred agent schedule remove    Remove a scheduled task

alfred tools list               List all available tools
alfred tools list <agent>       List tools for a specific agent

alfred discord setup            Configure Discord bot
alfred discord start            Start the Discord bot
alfred discord status           Show Discord configuration
```

## Architecture

```
User / Discord
      |
      v
  Agent.run(message)
      |
      +---> Memory search (auto-recall relevant context)
      +---> Build system prompt (workspace files + memories)
      +---> LLM call (with tools available)
      |         |
      |         +---> Tool call? Execute, feed result back, loop
      |
      +---> Return response
      +---> Store new memories (optional)
```

### Project Structure

```
alfred-ai/
  alfred              CLI launcher (bash wrapper)
  __main__.py         CLI entry point — all commands route through here
  core/
    agent.py          Agent loop: perceive -> remember -> think -> act -> learn
    config.py         Config loading/saving (alfred.json)
    llm.py            Multi-provider LLM client (Anthropic, xAI, etc.)
    memory.py         Vector memory store (LanceDB + hybrid search)
    embeddings.py     Embedding model management
    tools.py          Tool registry, execution, and builtin tools
    tool_meta.py      Meta-tools (agents managing their own tools)
    tool_discovery.py Auto-discovery of shared tools
    workspace.py      Workspace creation and management
    models.py         Model registry and provider catalogs
    scheduler.py      Cron-based task scheduling
    discord.py        Discord bot — channel-to-agent routing
  cli/
    setup.py          Interactive setup wizard
  models/
    base.py           Base model definitions
    trade.py          Trading model definitions
    social.py         Social model definitions
  tools/
    web_search.py     Web search tool
```

### Key Concepts

**Agents** are the core unit. Each agent has its own workspace directory containing:
- `SOUL.md` — identity, personality, system prompt
- `USER.md` — what the agent knows about its user
- `AGENTS.md` — awareness of other agents
- `TOOLS.md` — tool documentation and usage notes

**Memory** is automatic. When an agent receives a message, it searches its vector store for relevant past context and injects it into the prompt. Hybrid search combines vector similarity (0.7 weight) with text matching (0.3 weight).

**Tools** use a layered discovery system:
1. **Builtin** — memory read/write, always available
2. **Shared** — in the `tools/` directory, available to all agents
3. **Workspace** — in an agent's `workspace/tools/` directory, private to that agent
4. **Meta-tools** — agents can create, edit, and manage their own tools at runtime

**Discord integration** maps channels to agents. Each channel gets its own agent instance for thread safety. Thread messages inherit their parent channel's agent. Configure with `alfred discord setup` and run with `alfred discord start`.

## Configuration

All configuration lives in `alfred.json` (auto-generated by `alfred setup`):

```json
{
  "llm": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929"
  },
  "providers": {
    "anthropic": {
      "api_key": "sk-ant-...",
      "model": "claude-sonnet-4-5-20250929"
    }
  },
  "agents": {
    "alfred": {
      "workspace": "workspaces/alfred",
      "description": "General-purpose assistant",
      "status": "active"
    }
  },
  "discord": {
    "bot_token": "...",
    "guild_id": "...",
    "channels": {
      "CHANNEL_ID": {
        "name": "general",
        "agent": "alfred",
        "require_mention": false
      }
    }
  }
}
```

> `alfred.json` contains API keys and tokens — it's excluded from git by default.

## Discord Setup

1. Create a bot at [discord.com/developers](https://discord.com/developers/applications)
2. Enable the **Message Content** intent
3. Invite the bot to your server with message read/write permissions
4. Run `alfred discord setup` — it auto-discovers your server and channels
5. Map each channel to an agent
6. Run `alfred discord start`

The bot runs in the foreground. Use `tmux`, `screen`, or `nohup` for background operation.

## Requirements

- Python 3.10+
- An API key from a supported LLM provider (Anthropic, xAI)

## License

MIT
