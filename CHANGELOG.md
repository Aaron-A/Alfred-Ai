# Changelog

All notable changes to Alfred AI will be documented in this file.

## [0.1.0] - 2026-02-22

### Added
- **Agent framework** with persistent memory, session management, and multi-provider LLM support (Anthropic, xAI, OpenAI, Ollama)
- **Vector memory** powered by LanceDB with hybrid search, importance scoring, temporal decay, deduplication, and weekly compaction
- **Self-learning** via optional post-session reflection with confidence scoring and outcome linking
- **Tool system** with auto-discovery, auto-retry, result summarization, and meta-tools
- **Multi-agent delegation** with `delegate_to` and async `send_message` inbox
- **Web dashboard** at `localhost:7700` with agents, schedules, metrics, and trading views
- **Interactive architecture diagram** at `localhost:7700/architecture`
- **Multi-agent trading dashboard** with per-agent tabs, live charts, and IBKR Pro (Fixed) commission tracking
- **Trading bots** for BTC/USD (crypto) and TSLA (stocks) via Alpaca API
- **Discord integration** with channel-to-agent mapping and scheduled task posting
- **Scheduling** with cron expressions, retry logic, run history, and weekly maintenance
- **Cost tracking** with per-model USD estimates, daily alerts, and dashboard views
- **Alerting** via Discord webhooks for error spikes, cost thresholds, and bot crashes
- **HTTP API** with REST endpoints, SSE streaming, and webhook triggers
- **CLI** with 60+ commands for agent management, scheduling, sessions, and Discord
- **Session persistence** with full tool chain preservation and sliding window
- **`pyproject.toml`** with `uv` support for dependency management
