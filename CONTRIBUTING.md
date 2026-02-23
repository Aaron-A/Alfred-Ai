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

**Quick scaffold** — agents can create their own workspace tools at runtime:
```bash
alfred tool create my_tool       # Scaffolds a new tool file
alfred tool search "web scraper" # Check if one already exists
```

**Manual creation** — for shared tools available to all agents:

1. Create `tools/your_tool.py` (shared) or `workspaces/<agent>/tools/your_tool.py` (agent-specific)
2. Use `snake_case` for filenames and function names
3. Add full type hints to all parameters — these become the tool's input schema
4. Include a clear docstring — the first line becomes the tool description shown to the LLM
5. Handle errors gracefully — return error strings, don't raise exceptions
6. Add a `register(registry)` function that calls `registry.register_function()`
7. Optionally add `TOOL_META` dict with `version`, `author`, and `description`
8. The tool auto-discovers on next startup

```python
TOOL_META = {"version": "1.0.0", "author": "you", "description": "Does something useful"}

def my_tool(query: str, limit: int = 10) -> str:
    """Search for things and return results."""
    try:
        # your logic here
        return f"Found {len(results)} results"
    except Exception as e:
        return f"Error: {e}"

def register(registry):
    registry.register_function("my_tool", my_tool)
```

## Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- `alfred status` output (redact any API keys)

## License

By contributing, you agree to the [MIT License](LICENSE).
