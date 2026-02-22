# Contributing to Alfred AI

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/Aaron-A/Alfred-Ai.git
cd Alfred-Ai
uv sync --all-extras
sudo ln -sf "$(pwd)/alfred" /usr/local/bin/alfred
alfred setup
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Test locally: `alfred start --fg`
4. Commit with a clear message describing the *why*, not just the *what*
5. Push and open a pull request

## Project Structure

- `core/` — agent framework (agent loop, LLM client, memory, tools, API)
- `tools/` — shared tools available to all agents
- `models/` — Pydantic model definitions
- `static/` — web dashboard (HTML/CSS/JS)
- `cli/` — CLI setup wizard
- `workspaces/` — per-agent persistent storage (gitignored)

## Conventions

- Python 3.10+ type hints (use `list[str]` not `List[str]`)
- Logging via `core.logging.get_logger(__name__)`
- Tools: every `.py` file in a `tools/` directory must export a `register(registry: ToolRegistry)` function
- Config: all runtime config lives in `alfred.json` (gitignored, never committed)
- No secrets in code — API keys go in `alfred.json` or `.env`

## Adding a Tool

1. Create `tools/your_tool.py`
2. Define your function(s)
3. Add a `register(registry)` function that calls `registry.register_function()`
4. The tool auto-discovers on next startup

## Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- `alfred status` output (redact any API keys)

## License

By contributing, you agree to the [MIT License](LICENSE).
