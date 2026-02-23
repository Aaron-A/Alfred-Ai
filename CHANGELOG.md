# Changelog

All notable changes to Alfred AI will be documented in this file.

## [0.1.2] - 2026-02-23

### Added
- Context Window Guard — monitors context size before each API call during tool loops and compacts older rounds when approaching the budget threshold (default 60% of model's context window), preventing quadratic token growth on long runs
- `schedule_max_tool_rounds` config — separate (lower) tool round cap for scheduled runs (default 15)
- Per-agent `context_window_tokens`, `context_budget_pct`, and `context_reserve_tokens` config options

## [0.1.1] - 2026-02-22

### Added
- Circuit breaker for failing tools (auto-skip after 3 consecutive failures, resets on success)
- Per-agent daily cost budgets (`max_daily_cost`) — agent refuses to run if exceeded
- Secret sanitization — redacts API keys and tokens (sk-, xai-, ghp_, AKIA, etc.) in agent responses
- Schedule auto-disable after 5 consecutive failures with Discord/webhook alert
- Delegation timeout (default 5 minutes) prevents hung delegated tasks
- Memory quotas with auto-compaction at 10K memories per agent
- Generic webhook alerts (`alerts.webhook_url`) alongside Discord
- Tool execution timing and logging per call
- Forced reflection for scheduled task runs
- Session cleanup (>30 days) and metrics cleanup (>90 days) in maintenance cron
- Model pricing fallbacks for unknown model variants (prefix-based lookup)
- Message delivery confirmation with unique IDs in `send_message`
- Agent templates for rapid creation (trader, social, research, etc.)
- Tool management CLI: `alfred tool info/install/create/search/remove`
- Service management CLI: `alfred service add/list/remove`
- One-line install script (`install.sh`)
- Migration CLI: `alfred migrate`, `alfred export`, `alfred import` for moving between directories or machines

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
