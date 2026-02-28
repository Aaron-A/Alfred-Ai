"""
Alfred AI - HTTP API Server
FastAPI endpoints for interacting with Alfred agents remotely.

Start with: alfred api start
Or:         alfred api start --port 8080

Endpoints:
    POST /v1/chat              Send a message to an agent, get response
    POST /v1/chat/stream       SSE streaming response
    POST /v1/memory/search     Search agent memory
    POST /v1/memory/store      Store a new memory
    GET  /v1/agents            List all agents
    POST /v1/agents            Create a new agent with workspace
    DELETE /v1/agents/{name}   Delete an agent, workspace, and Discord mapping
    GET  /v1/agents/{name}     Get agent details + session info
    POST /v1/agents/{name}/reset  Reset an agent's session
    GET  /v1/sessions/{agent}  List all saved sessions for an agent
    GET  /v1/sessions/{agent}/{id}  Get session messages
    DELETE /v1/sessions/{agent}/{id}  Delete a session
    GET  /v1/sessions/{agent}/{id}/export  Export session as markdown/text
    GET  /v1/agents/{name}/schedules  List schedules for an agent
    POST /v1/webhook/{agent}   Fire a webhook event
    GET  /v1/status            System status (providers, services, etc.)
    GET  /health               Health check
    GET  /                     Web dashboard
"""

import os
import re
import json
import time
import shutil
import asyncio
import queue
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import config, _load_config, CONFIG_FILE
from .agent import Agent, AgentConfig, AgentManager
from .logging import metrics, setup_logging, get_logger

logger = get_logger("api")

# ─── Curated Provider/Model Registry ──────────────────────────
# Shared by API endpoints, dashboard dropdowns, and agent switch_model tool.

PROVIDER_MODELS = {
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-0-20250918",
        "claude-haiku-3-5-20241022",
    ],
    "xai": [
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast",
        "grok-3-fast",
        "grok-3-mini-fast",
    ],
    "openai": [
        "gpt-5.2",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o3",
        "o4-mini",
    ],
    "ollama": [
        "llama3.1:8b",
        "llama3.1",
        "llama3.2",
        "mistral",
        "codellama",
        "deepseek-r1",
        "qwen2.5",
    ],
}


# ─── Pydantic Models ────────────────────────────────────────

class ChatRequest(BaseModel):
    agent: str = Field(default="alfred", description="Agent name")
    message: str = Field(..., description="Message to send")
    session_id: Optional[str] = Field(default=None, description="Session scope (default: 'api')")
    reset: bool = Field(default=False, description="Reset session before this message")

class ChatResponse(BaseModel):
    agent: str
    response: str
    session_id: str
    turns: int
    elapsed_ms: int
    tokens: Optional[dict] = None  # {"input": N, "output": N, "total": N}

class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    memory_type: Optional[str] = Field(default=None, description="Filter by type")
    top_k: int = Field(default=5, ge=1, le=50)
    agent_id: Optional[str] = Field(default=None, description="Filter by agent")

class MemoryStoreRequest(BaseModel):
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(default="generic", description="Memory type")
    tags: str = Field(default="", description="Comma-separated tags")
    agent_id: Optional[str] = Field(default=None, description="Agent that owns this memory")

class MemoryResponse(BaseModel):
    id: str
    message: str


# ─── Agent Pool ──────────────────────────────────────────────

class AgentPool:
    """
    Thread-safe pool of agent instances, keyed by (agent_name, session_id).
    Reuses agents across requests for session continuity.
    """

    def __init__(self):
        self._agents: dict[tuple[str, str], Agent] = {}
        self._lock = asyncio.Lock()

    async def get(self, agent_name: str, session_id: str = "api") -> Agent:
        """Get or create an agent for the given name + session scope."""
        key = (agent_name, session_id)

        async with self._lock:
            if key not in self._agents:
                # Validate agent exists in config
                cfg = _load_config()
                agents_cfg = cfg.get("agents", {})

                if agent_name not in agents_cfg:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Agent '{agent_name}' not found. Available: {list(agents_cfg.keys())}"
                    )

                agent_data = dict(agents_cfg[agent_name])
                agent_data["name"] = agent_name

                # Check status
                if agent_data.get("status") == "paused":
                    raise HTTPException(status_code=409, detail=f"Agent '{agent_name}' is paused.")

                # Resolve workspace
                workspace = Path(agent_data.get("workspace", f"workspaces/{agent_name}"))
                if not workspace.is_absolute():
                    workspace = config.PROJECT_ROOT / workspace
                agent_data["workspace"] = str(workspace)

                # Ensure workspace dirs
                workspace.mkdir(parents=True, exist_ok=True)
                (workspace / "memory").mkdir(exist_ok=True)
                (workspace / "tools").mkdir(exist_ok=True)

                agent_config = AgentConfig.from_dict(agent_data)
                agent = Agent(agent_config, session_id=session_id)
                self._agents[key] = agent

            return self._agents[key]

    async def reset(self, agent_name: str, session_id: str = "api"):
        """Reset an agent's session."""
        key = (agent_name, session_id)
        async with self._lock:
            if key in self._agents:
                self._agents[key].reset()
            # Also reset even if not loaded (delete session file)
            cfg = _load_config()
            agent_data = cfg.get("agents", {}).get(agent_name, {})
            workspace = Path(agent_data.get("workspace", f"workspaces/{agent_name}"))
            if not workspace.is_absolute():
                workspace = config.PROJECT_ROOT / workspace
            session_file = workspace / (
                "session.json" if session_id == "cli"
                else f"session_{session_id}.json"
            )
            session_file.unlink(missing_ok=True)


# ─── App Factory ─────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Alfred AI",
        description="Memory-first AI agent framework",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # CORS — allow everything for local dev, lock down in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    pool = AgentPool()

    # ─── Dashboard ────────────────────────────────────────

    static_dir = Path(__file__).parent.parent / "static"

    @app.get("/", include_in_schema=False)
    async def dashboard():
        """Serve the web dashboard."""
        html_file = static_dir / "dashboard.html"
        if html_file.exists():
            return FileResponse(str(html_file), media_type="text/html")
        return {"message": "Alfred AI API", "docs": "/docs"}

    @app.get("/architecture", include_in_schema=False)
    async def architecture():
        """Serve the architecture diagram page."""
        html_file = static_dir / "architecture.html"
        if html_file.exists():
            return FileResponse(str(html_file), media_type="text/html")
        return {"error": "Architecture page not found"}

    # ─── Health ──────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    # ─── Status ──────────────────────────────────────────

    @app.get("/v1/status")
    async def status():
        if not CONFIG_FILE.exists():
            raise HTTPException(status_code=503, detail="Alfred not configured. Run: alfred setup")

        cfg = _load_config()
        llm = cfg.get("llm", {})
        primary = llm.get("primary", {})
        if not primary and "provider" in llm:
            primary = {"provider": llm["provider"], "model": llm.get("model", "?")}

        secondary = llm.get("secondary", {})

        # Check Discord (with health check, not just PID)
        discord_cfg = cfg.get("discord", {})
        discord_info = {
            "configured": bool(discord_cfg.get("bot_token")),
            "running": False,
            "healthy": False,
            "channels": len(discord_cfg.get("channels", {})),
            "health_message": "",
        }
        if discord_cfg.get("bot_token"):
            try:
                from .discord import is_bot_running, is_bot_healthy
                discord_info["running"] = is_bot_running() is not None
                healthy, msg = is_bot_healthy()
                discord_info["healthy"] = healthy
                discord_info["health_message"] = msg
            except Exception:
                discord_info["health_message"] = "status check failed"

        return {
            "version": cfg.get("version", "0.1.0"),
            "timezone": config.TIMEZONE,
            "server_time": datetime.now(config.tz).isoformat(),
            "primary_llm": primary or None,
            "secondary_llm": secondary or None,
            "providers": list(cfg.get("providers", {}).keys()),
            "agents": list(cfg.get("agents", {}).keys()),
            "discord": discord_info,
            "embeddings": cfg.get("embeddings", {}),
        }

    # ─── Chat ────────────────────────────────────────────

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        session_id = req.session_id or "api"

        agent = await pool.get(req.agent, session_id)

        if req.reset:
            agent.reset()

        # Run agent in thread pool (it's synchronous — LLM calls block)
        loop = asyncio.get_event_loop()
        start = time.monotonic()
        try:
            response = await loop.run_in_executor(None, agent.run, req.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        elapsed = int((time.monotonic() - start) * 1000)

        # Get latest token usage from metrics (delta since this request)
        agent_metrics = metrics.summary(req.agent)
        token_info = {
            "input": agent_metrics.get("input_tokens", 0),
            "output": agent_metrics.get("output_tokens", 0),
            "total": agent_metrics.get("total_tokens", 0),
        }

        return ChatResponse(
            agent=req.agent,
            response=response,
            session_id=session_id,
            turns=agent.session_info["turns"],
            elapsed_ms=elapsed,
            tokens=token_info,
        )

    # ─── Streaming Chat ─────────────────────────────────

    @app.post("/v1/chat/stream")
    async def chat_stream(req: ChatRequest):
        """
        Stream a response from an agent using Server-Sent Events (SSE).

        NOTE: Streaming bypasses the tool loop — the LLM generates a direct
        text response. For tool-enabled interactions, use /v1/chat instead.

        Returns an SSE stream with:
        - data: {"chunk": "text"} for each text chunk
        - data: {"done": true, "agent": "name"} when complete
        """
        session_id = req.session_id or "api"
        agent = await pool.get(req.agent, session_id)

        if req.reset:
            agent.reset()

        async def generate():
            """SSE generator — runs agent streaming in thread pool."""
            q = queue.Queue()
            loop = asyncio.get_event_loop()

            def _stream_worker():
                """Run in thread pool, push chunks to queue."""
                try:
                    for chunk in agent.run_stream(req.message):
                        q.put(("chunk", chunk))
                    q.put(("done", None))
                except Exception as e:
                    q.put(("error", str(e)))

            # Start streaming in background thread
            fut = loop.run_in_executor(None, _stream_worker)

            while True:
                try:
                    # Non-blocking poll with short timeout
                    msg_type, msg_data = await loop.run_in_executor(
                        None, lambda: q.get(timeout=0.1)
                    )

                    if msg_type == "chunk":
                        yield f"data: {json.dumps({'chunk': msg_data})}\n\n"
                    elif msg_type == "done":
                        yield f"data: {json.dumps({'done': True, 'agent': req.agent})}\n\n"
                        break
                    elif msg_type == "error":
                        yield f"data: {json.dumps({'error': msg_data})}\n\n"
                        break
                except queue.Empty:
                    # Check if the worker is done
                    if fut.done():
                        break
                    continue

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ─── Agents ──────────────────────────────────────────

    @app.get("/v1/agents")
    async def list_agents():
        cfg = _load_config()
        agents = cfg.get("agents", {})
        # Resolve global defaults for agents without per-agent overrides
        global_provider = cfg.get("llm", {}).get("primary", {}).get("provider",
                          cfg.get("llm", {}).get("provider", "anthropic"))
        global_model = cfg.get("llm", {}).get("primary", {}).get("model",
                       cfg.get("llm", {}).get("model", ""))
        result = []
        for name, acfg in agents.items():
            result.append({
                "name": name,
                "description": acfg.get("description", ""),
                "status": acfg.get("status", "active"),
                "workspace": acfg.get("workspace", ""),
                "provider": acfg.get("provider", global_provider),
                "model": acfg.get("model", global_model),
                # Config fields for the edit modal
                "max_tool_rounds": acfg.get("max_tool_rounds", 10),
                "max_daily_cost": acfg.get("max_daily_cost", 0),
                "context_budget_pct": acfg.get("context_budget_pct", 0.60),
                "schedule_max_tool_rounds": acfg.get("schedule_max_tool_rounds", 15),
                "schedule_run_mode": acfg.get("schedule_run_mode", "tool_use"),
            })
        return {"agents": result}

    @app.get("/v1/agents/{name}")
    async def get_agent(name: str, session_id: str = "api"):
        cfg = _load_config()
        agents = cfg.get("agents", {})
        if name not in agents:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found.")

        acfg = agents[name]

        # Try to get session info if agent is loaded
        session = {}
        try:
            agent = await pool.get(name, session_id)
            session = agent.session_info
        except HTTPException:
            pass

        return {
            "name": name,
            "description": acfg.get("description", ""),
            "status": acfg.get("status", "active"),
            "workspace": acfg.get("workspace", ""),
            "birthday": acfg.get("birthday", ""),
            "creator": acfg.get("creator", ""),
            "session": session,
        }

    @app.post("/v1/agents/{name}/reset")
    async def reset_agent(name: str, session_id: str = "api"):
        await pool.reset(name, session_id)
        return {"message": f"Session reset for {name} (session: {session_id})"}

    @app.get("/v1/agents/{name}/schedules")
    async def get_agent_schedules_endpoint(name: str):
        """List all schedules for an agent with computed fields."""
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})

        if name not in agents_cfg:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found.")

        from .scheduler import get_agent_schedules, describe_cron, next_run

        schedules = get_agent_schedules(name)
        result = []
        for s in schedules:
            d = s.to_dict()
            d["human_schedule"] = describe_cron(s.cron)
            nxt = next_run(s.cron) if s.enabled else None
            d["next_run"] = nxt.isoformat() if nxt else None
            d["success_rate"] = round(s.success_rate, 1)
            result.append(d)

        return {"agent": name, "schedules": result}

    @app.post("/v1/agents/{name}/schedules/{schedule_id}/run")
    async def trigger_schedule_now(name: str, schedule_id: str):
        """Manually trigger a scheduled task to run immediately."""
        cfg = _load_config()
        if name not in cfg.get("agents", {}):
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found.")

        scheduler = getattr(app.state, "scheduler", None)
        if not scheduler:
            raise HTTPException(status_code=503, detail="Scheduler not available.")

        result = scheduler.trigger_now(name, schedule_id)

        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found.")
        if result["status"] == "no_runner":
            raise HTTPException(status_code=503, detail="No agent runner configured.")
        if result["status"] == "already_running":
            raise HTTPException(status_code=409, detail=f"Schedule '{schedule_id}' is already running.")

        return result

    @app.get("/v1/agents/{name}/schedules/{schedule_id}/status")
    async def get_schedule_run_status(name: str, schedule_id: str):
        """Check if a specific schedule is currently running."""
        scheduler = getattr(app.state, "scheduler", None)
        running = scheduler.is_schedule_running(schedule_id) if scheduler else False
        return {"agent": name, "schedule_id": schedule_id, "running": running}

    @app.patch("/v1/agents/{name}/schedules/{schedule_id}")
    async def toggle_schedule_endpoint(name: str, schedule_id: str, body: dict):
        """Toggle a schedule's enabled state (pause/resume)."""
        from .scheduler import toggle_schedule
        enabled = body.get("enabled")
        if enabled is None:
            raise HTTPException(status_code=400, detail="Missing 'enabled' field")
        ok = toggle_schedule(name, schedule_id, bool(enabled))
        if not ok:
            raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found")
        action = "enabled" if enabled else "disabled"
        logger.info(f"Schedule '{schedule_id}' for agent '{name}' {action} via API")
        return {"message": f"Schedule '{schedule_id}' {action}"}

    @app.delete("/v1/agents/{name}/schedules/{schedule_id}")
    async def delete_schedule_endpoint(name: str, schedule_id: str):
        """Permanently delete a schedule."""
        from .scheduler import remove_schedule
        ok = remove_schedule(name, schedule_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found")
        logger.info(f"Schedule '{schedule_id}' deleted from agent '{name}' via API")
        return {"message": f"Schedule '{schedule_id}' deleted"}

    # ─── Sessions ────────────────────────────────────────

    @app.get("/v1/sessions/{agent_name}")
    async def list_sessions(agent_name: str):
        """List all saved sessions for an agent."""
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})

        if agent_name not in agents_cfg:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

        workspace = Path(agents_cfg[agent_name].get("workspace", f"workspaces/{agent_name}"))
        if not workspace.is_absolute():
            workspace = config.PROJECT_ROOT / workspace

        sessions = Agent.list_sessions(str(workspace))
        return {"agent": agent_name, "sessions": sessions}

    @app.get("/v1/sessions/{agent_name}/{session_id}")
    async def get_session(agent_name: str, session_id: str, last: int = None):
        """Get messages from a specific session. Use ?last=N for last N turns."""
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})

        if agent_name not in agents_cfg:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

        workspace = Path(agents_cfg[agent_name].get("workspace", f"workspaces/{agent_name}"))
        if not workspace.is_absolute():
            workspace = config.PROJECT_ROOT / workspace

        messages = Agent.get_session_messages(str(workspace), session_id)
        if not messages:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        if last:
            messages = messages[-(last * 2):]

        return {
            "agent": agent_name,
            "session_id": session_id,
            "turns": len(messages) // 2,
            "messages": messages,
        }

    @app.delete("/v1/sessions/{agent_name}/{session_id}")
    async def delete_session(agent_name: str, session_id: str):
        """Delete a saved session."""
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})

        if agent_name not in agents_cfg:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

        workspace = Path(agents_cfg[agent_name].get("workspace", f"workspaces/{agent_name}"))
        if not workspace.is_absolute():
            workspace = config.PROJECT_ROOT / workspace

        # Delete the file first (before pool.reset which also unlinks)
        deleted = Agent.delete_session(str(workspace), session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        # Evict from pool so stale in-memory agent is cleared
        await pool.reset(agent_name, session_id)

        return {"message": f"Session '{session_id}' deleted for {agent_name}."}

    @app.get("/v1/sessions/{agent_name}/{session_id}/export")
    async def export_session(agent_name: str, session_id: str, format: str = "markdown"):
        """Export a session as formatted text. Accepts ?format=markdown or ?format=text."""
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})

        if agent_name not in agents_cfg:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

        workspace = Path(agents_cfg[agent_name].get("workspace", f"workspaces/{agent_name}"))
        if not workspace.is_absolute():
            workspace = config.PROJECT_ROOT / workspace

        content = Agent.export_session(str(workspace), session_id, format=format)
        if not content:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        media_type = "text/markdown" if format == "markdown" else "text/plain"
        return StreamingResponse(
            iter([content]),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={agent_name}_{session_id}.{'md' if format == 'markdown' else 'txt'}"}
        )

    # ─── Memory ──────────────────────────────────────────

    @app.post("/v1/memory/search")
    async def memory_search(req: MemorySearchRequest):
        from .memory import MemoryStore

        loop = asyncio.get_event_loop()

        def _search():
            aid = req.agent_id or "default"
            store = MemoryStore(agent_id=aid)
            # Scope search to the requesting agent's memories
            where = f"agent_id = '{aid}'" if req.agent_id else None
            return store.search(
                query=req.query,
                memory_type=req.memory_type,
                top_k=req.top_k,
                where=where,
            )

        try:
            results = await loop.run_in_executor(None, _search)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Clean up results (remove vectors, they're huge)
        clean = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "vector"}
            clean.append(entry)

        return {"results": clean, "count": len(clean)}

    @app.post("/v1/memory/store", response_model=MemoryResponse)
    async def memory_store(req: MemoryStoreRequest):
        from .memory import MemoryStore
        from models.base import MemoryRecord

        loop = asyncio.get_event_loop()

        def _store():
            aid = req.agent_id or "default"
            store = MemoryStore(agent_id=aid)
            record = MemoryRecord(
                content=req.content,
                memory_type=req.memory_type,
                tags=req.tags,
                agent_id=aid,
            )
            return store.store(record)

        try:
            record_id = await loop.run_in_executor(None, _store)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return MemoryResponse(id=record_id, message="Memory stored")

    # ─── Metrics ──────────────────────────────────────────

    @app.get("/v1/metrics")
    async def get_metrics(agent: str = None, period: str = "day"):
        """
        Get agent activity metrics.

        Args:
            agent: Filter to a specific agent
            period: 5m, 15m, 30m, 1h, 12h, day, week, month, year, all, session
        """
        if period == "session":
            # Existing behavior — current in-memory snapshot
            if agent:
                return {"agent": agent, "metrics": metrics.summary(agent)}
            return metrics.snapshot()

        # Time-series query from SQLite
        from .metrics_store import MetricsStore
        store = MetricsStore()
        return store.query(period=period, agent=agent)

    @app.get("/v1/vector-queries")
    async def get_vector_queries(agent: str = None, period: str = "day", limit: int = 50):
        """
        Get vector search query logs for observability.

        Shows what the agents are searching for, how many results come back,
        and average relevance scores.
        """
        import sqlite3
        from .metrics_store import _PERIOD_OFFSETS, MetricsStore

        offset = _PERIOD_OFFSETS.get(period, _PERIOD_OFFSETS["day"])
        cutoff = MetricsStore._compute_cutoff(datetime.now(config.tz), offset)
        db_path = config.DATA_DIR / "metrics.db"

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            if agent:
                rows = conn.execute(
                    """SELECT * FROM vector_queries
                       WHERE timestamp >= ? AND agent = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (cutoff, agent, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM vector_queries
                       WHERE timestamp >= ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (cutoff, limit),
                ).fetchall()

            conn.close()
            return {
                "queries": [dict(r) for r in rows],
                "count": len(rows),
                "period": period,
            }
        except Exception as e:
            return {"queries": [], "count": 0, "error": str(e)}

    # ─── Webhooks ─────────────────────────────────────────

    @app.post("/v1/webhook/{agent_name}")
    async def webhook(agent_name: str, payload: dict = {}):
        """
        Receive an external event and route it to an agent.

        The agent processes the event as a message and returns a response.
        Useful for: CI/CD notifications, monitoring alerts, cron triggers,
        IoT events, or any external service that can POST JSON.

        Optional security: Set "webhook_secret" in agent config.
        Pass it via X-Webhook-Secret header or "secret" field in body.

        Payload format:
            {
                "event": "deploy",           # Event type (optional, for filtering)
                "message": "Deploy complete", # Message to send to agent
                "data": {...},               # Additional data (injected as context)
                "secret": "...",             # Auth secret (optional)
                "respond": true              # If false, fire-and-forget (default: true)
            }
        """
        from fastapi import Request

        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})

        if agent_name not in agents_cfg:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )

        agent_data = agents_cfg[agent_name]

        # Check webhook secret if configured
        webhook_secret = agent_data.get("webhook_secret", "")
        if webhook_secret:
            provided_secret = payload.get("secret", "")
            if provided_secret != webhook_secret:
                raise HTTPException(status_code=403, detail="Invalid webhook secret")

        # Build message from payload
        event_type = payload.get("event", "webhook")
        message = payload.get("message", "")
        data = payload.get("data", {})
        respond = payload.get("respond", True)

        if not message and data:
            message = f"Webhook event: {event_type}\nData: {json.dumps(data, indent=2)}"
        elif not message:
            message = f"Webhook event: {event_type}"

        # Prefix with event context
        full_message = f"[Webhook: {event_type}] {message}"

        logger.info(f"Webhook -> {agent_name}: {event_type} ({len(message)} chars)")

        if not respond:
            # Fire-and-forget: run in background, don't wait for response
            async def _bg_run():
                try:
                    agent = await pool.get(agent_name, "webhook")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, agent.run, full_message)
                except Exception as e:
                    logger.error(f"Webhook background task failed: {e}")

            asyncio.create_task(_bg_run())
            return {
                "status": "accepted",
                "agent": agent_name,
                "event": event_type,
            }

        # Synchronous: run and return response
        try:
            agent = await pool.get(agent_name, "webhook")
            loop = asyncio.get_event_loop()
            start = time.monotonic()
            response = await loop.run_in_executor(None, agent.run, full_message)
            elapsed = int((time.monotonic() - start) * 1000)

            return {
                "status": "ok",
                "agent": agent_name,
                "event": event_type,
                "response": response,
                "elapsed_ms": elapsed,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ─── Agent Config ─────────────────────────────────────

    @app.get("/v1/providers")
    async def list_providers():
        """Return the curated provider → models registry for UI dropdowns."""
        return {"providers": PROVIDER_MODELS}

    # ─── Create Agent ─────────────────────────────────────

    class AgentCreateRequest(BaseModel):
        name: str = Field(..., description="Agent name (lowercase, a-z, 0-9, hyphens)")
        description: str = Field(..., description="Short agent description")
        provider: Optional[str] = Field(default=None, description="LLM provider")
        model: Optional[str] = Field(default=None, description="LLM model")
        soul_prompt: Optional[str] = Field(default=None, description="Freeform prompt to LLM-generate SOUL.md")
        discord_channel_id: Optional[str] = Field(default=None, description="Discord channel ID to auto-link")

    def _validate_agent_name(name: str) -> str:
        """Validate agent name. Returns error message or empty string."""
        if not name:
            return "Agent name is required."
        if not re.match(r'^[a-z][a-z0-9-]{0,29}$', name):
            return "Name must be lowercase, start with a letter, only a-z/0-9/hyphens, max 30 chars."
        return ""

    async def _generate_soul_md(
        agent_name: str, description: str, soul_prompt: str,
        creator: str, birthday: str,
    ) -> str:
        """Use the primary LLM to generate a rich SOUL.md from the user's prompt."""
        from .llm import LLMClient

        system = (
            "You are an expert AI agent architect. Generate a SOUL.md file for a new AI agent.\n\n"
            "The SOUL.md defines the agent's identity, personality, operating rules, and boundaries.\n"
            "It is written in Markdown and loaded as the agent's core system prompt.\n\n"
            "Use this structure (adapt sections based on the agent's purpose):\n\n"
            "# SOUL.md - Who You Are\n\n"
            "You are {name}, an AI agent created by {creator}.\n"
            "Your birthday is {birthday} — the day you were first brought online.\n\n"
            "_One-line tagline capturing the agent's essence._\n\n"
            "## Core Truths\n(3-5 bold principles this agent lives by)\n\n"
            "## Boundaries\n(Hard rules the agent must never break, as a bullet list)\n\n"
            "## Personality\n(5-7 personality traits as bullet points)\n\n"
            "## Rules\n(Operational rules — check memory, log decisions, etc.)\n\n"
            "Key principles:\n"
            "- Be specific and opinionated, not generic\n"
            "- Include concrete boundaries with numbers where relevant\n"
            "- Write in second person (\"You are...\", \"You never...\")\n"
            "- The tone should match the agent's personality\n"
            "- Include a tagline in italics after the opening description\n"
            "- Reference memory tools for continuity\n"
            "- Do NOT include sections about tools or other agents (those are auto-generated separately)\n"
            "- Output ONLY the Markdown content, no code fences or explanations"
        )

        prompt = (
            f"Create SOUL.md for this agent:\n\n"
            f"**Name:** {agent_name}\n"
            f"**Description:** {description}\n"
            f"**Creator:** {creator}\n"
            f"**Birthday:** {birthday}\n\n"
            f"**User's vision for this agent:**\n{soul_prompt}"
        )

        loop = asyncio.get_event_loop()

        def _call_llm():
            client = LLMClient()
            return client.ask(prompt, system=system, max_tokens=2000, temperature=0.7)

        result = await loop.run_in_executor(None, _call_llm)

        # Strip code fences if the LLM wraps the output
        if result and result.strip().startswith("```"):
            lines = result.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            result = "\n".join(lines)

        return result

    @app.post("/v1/agents")
    async def create_agent(req: AgentCreateRequest):
        """
        Create a new agent with workspace and optionally LLM-generated SOUL.md.

        If soul_prompt is provided, uses the primary LLM to generate a rich SOUL.md.
        Otherwise, uses the default template.
        """
        # 1. Validate name
        err = _validate_agent_name(req.name)
        if err:
            raise HTTPException(status_code=400, detail=err)

        # 2. Check name doesn't exist
        cfg = _load_config()
        if req.name in cfg.get("agents", {}):
            raise HTTPException(status_code=409, detail=f"Agent '{req.name}' already exists.")

        # 3. Validate provider/model if specified
        if req.provider:
            if req.provider not in PROVIDER_MODELS:
                raise HTTPException(status_code=400, detail=f"Unknown provider '{req.provider}'.")
            if req.model and req.model not in PROVIDER_MODELS[req.provider]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown model '{req.model}' for '{req.provider}'."
                )

        # 4. Resolve creator name from existing config
        creator = ""
        for acfg in cfg.get("agents", {}).values():
            if acfg.get("creator"):
                creator = acfg["creator"]
                break

        # 5. Generate SOUL.md if soul_prompt provided
        birthday = datetime.now().strftime("%B %d, %Y")
        soul_generated = False
        custom_soul = None

        if req.soul_prompt and req.soul_prompt.strip():
            try:
                custom_soul = await _generate_soul_md(
                    req.name, req.description, req.soul_prompt,
                    creator or "its creator", birthday
                )
                soul_generated = bool(custom_soul and custom_soul.strip())
            except Exception as e:
                logger.error(f"SOUL.md generation failed (falling back to template): {e}")

        # 6. Create workspace
        from .workspace import create_workspace
        from .tools import ToolRegistry, register_builtin_tools
        from .tool_discovery import discover_shared_tools

        workspace_path = str(config.PROJECT_ROOT / "workspaces" / req.name)

        loop = asyncio.get_event_loop()

        def _create_workspace():
            temp_registry = ToolRegistry()
            register_builtin_tools(temp_registry)
            discover_shared_tools(temp_registry)
            return create_workspace(
                workspace_path, req.name,
                registry=temp_registry,
                creator=creator,
                birthday=birthday,
                user_name=creator,
            )

        created_files = await loop.run_in_executor(None, _create_workspace)

        # 7. Overwrite SOUL.md if LLM-generated content exists
        if soul_generated and custom_soul:
            soul_path = Path(workspace_path) / "SOUL.md"
            soul_path.write_text(custom_soul.strip() + "\n")

        # 8. Save to alfred.json
        from .config import _save_config

        if "agents" not in cfg:
            cfg["agents"] = {}

        agent_entry = {
            "workspace": f"workspaces/{req.name}",
            "description": req.description,
            "status": "active",
            "birthday": datetime.now().strftime("%Y-%m-%d"),
            "creator": creator,
        }
        if req.provider:
            agent_entry["provider"] = req.provider
        if req.model:
            agent_entry["model"] = req.model

        cfg["agents"][req.name] = agent_entry

        # Add Discord channel mapping if provided
        if req.discord_channel_id and req.discord_channel_id.strip():
            channel_id = req.discord_channel_id.strip()
            if not channel_id.isdigit():
                raise HTTPException(status_code=400, detail="Discord channel ID must be numeric.")
            if "discord" not in cfg:
                cfg["discord"] = {}
            if "channels" not in cfg["discord"]:
                cfg["discord"]["channels"] = {}
            cfg["discord"]["channels"][channel_id] = {
                "name": req.name,
                "agent": req.name,
                "require_mention": False,
            }

        _save_config(cfg)

        logger.info(f"Created agent '{req.name}' (soul_generated={soul_generated})")

        return {
            "name": req.name,
            "workspace": f"workspaces/{req.name}",
            "description": req.description,
            "status": "active",
            "files_created": [str(f) for f in created_files],
            "soul_generated": soul_generated,
            "restart_required": True,
        }

    # ─── Delete Agent ────────────────────────────────────

    @app.delete("/v1/agents/{name}")
    async def delete_agent(name: str):
        """Delete an agent: remove config, workspace, Discord mapping, and pool entries."""
        from .config import _save_config
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})

        if name not in agents_cfg:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found.")

        if len(agents_cfg) <= 1:
            raise HTTPException(status_code=409, detail="Cannot delete the last remaining agent.")

        # 1. Remove agent entry
        del cfg["agents"][name]

        # 2. Remove Discord channel mappings for this agent
        discord_channels = cfg.get("discord", {}).get("channels", {})
        channels_removed = [
            ch_id for ch_id, ch_cfg in discord_channels.items()
            if ch_cfg.get("agent") == name
        ]
        for ch_id in channels_removed:
            del cfg["discord"]["channels"][ch_id]

        # 3. Save config
        _save_config(cfg)

        # 4. Purge metrics
        try:
            from .metrics_store import MetricsStore
            deleted_rows = MetricsStore().delete_by_agent(name)
            logger.info(f"Purged {deleted_rows} metric rows for agent '{name}'")
        except Exception as e:
            logger.warning(f"Metrics cleanup failed for '{name}': {e}")

        # 5. Purge memory vectors from all LanceDB tables
        try:
            from .memory import MemoryStore
            ms = MemoryStore()
            for table_name in ms.list_tables():
                tbl = ms.db.open_table(table_name)
                tbl.delete(f"agent_id = '{name}'")
        except Exception as e:
            logger.warning(f"Memory cleanup failed for '{name}': {e}")

        # 6. Delete workspace directory
        workspace_dir = config.PROJECT_ROOT / "workspaces" / name
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)

        # 7. Evict from agent pool
        async with pool._lock:
            keys_to_delete = [k for k in pool._agents if k[0] == name]
            for k in keys_to_delete:
                del pool._agents[k]

        logger.info(f"Deleted agent '{name}' (channels_removed={len(channels_removed)})")

        return {"deleted": name, "restart_required": True}

    # ─── Agent Config Update ──────────────────────────────

    class AgentConfigUpdate(BaseModel):
        provider: Optional[str] = None
        model: Optional[str] = None
        max_tool_rounds: Optional[int] = None
        max_daily_cost: Optional[float] = None
        context_budget_pct: Optional[float] = None
        schedule_max_tool_rounds: Optional[int] = None
        schedule_run_mode: Optional[str] = None
        temperature: Optional[float] = None

    @app.patch("/v1/agents/{name}/config")
    async def update_agent_config(name: str, update: AgentConfigUpdate):
        """
        Update an agent's config in alfred.json.

        Only accepts known providers and their known models.
        """
        from .config import _save_config

        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})

        if name not in agents_cfg:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found.")

        # Validate provider
        if update.provider and update.provider not in PROVIDER_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider '{update.provider}'. Known: {list(PROVIDER_MODELS.keys())}"
            )

        # Validate model belongs to provider
        if update.model and update.provider:
            known_models = PROVIDER_MODELS.get(update.provider, [])
            if update.model not in known_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown model '{update.model}' for provider '{update.provider}'. Known: {known_models}"
                )

        # Apply updates
        if update.provider:
            agents_cfg[name]["provider"] = update.provider
        if update.model:
            agents_cfg[name]["model"] = update.model
        if update.max_tool_rounds is not None:
            agents_cfg[name]["max_tool_rounds"] = max(1, min(50, update.max_tool_rounds))
        if update.max_daily_cost is not None:
            agents_cfg[name]["max_daily_cost"] = max(0, update.max_daily_cost)
        if update.context_budget_pct is not None:
            agents_cfg[name]["context_budget_pct"] = max(0.1, min(0.95, update.context_budget_pct))
        if update.schedule_max_tool_rounds is not None:
            agents_cfg[name]["schedule_max_tool_rounds"] = max(1, min(50, update.schedule_max_tool_rounds))
        if update.temperature is not None:
            agents_cfg[name]["temperature"] = max(0, min(2, update.temperature))
        if update.schedule_run_mode is not None:
            if update.schedule_run_mode not in ("tool_use", "structured", "batch"):
                raise HTTPException(status_code=400, detail="schedule_run_mode must be 'tool_use', 'structured', or 'batch'")
            agents_cfg[name]["schedule_run_mode"] = update.schedule_run_mode

        cfg["agents"] = agents_cfg
        _save_config(cfg)

        logger.info(f"Updated agent '{name}' config: {update.model_dump(exclude_none=True)}")

        return {
            "message": f"Agent '{name}' config updated",
            "provider": agents_cfg[name].get("provider"),
            "model": agents_cfg[name].get("model"),
            "restart_required": True,
        }

    # ─── Workspace File Viewer ──────────────────────────

    _FILE_WHITELIST = {"SOUL.md", "USER.md", "AGENTS.md", "TOOLS.md"}

    @app.get("/v1/agents/{name}/files/{filepath:path}")
    async def get_agent_file(name: str, filepath: str):
        """Read a workspace file for the given agent.

        Allowed paths: SOUL.md, USER.md, AGENTS.md, TOOLS.md,
        memory/<date>.md, memory/sessions/<snapshot>.md
        """
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        if name not in agents_cfg:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

        workspace = Path(agents_cfg[name].get("workspace", f"workspaces/{name}"))
        if not workspace.is_absolute():
            workspace = config.PROJECT_ROOT / workspace

        # Sanitise: resolve and ensure it stays inside workspace
        target = (workspace / filepath).resolve()
        ws_resolved = str(workspace.resolve()) + "/"
        if not str(target).startswith(ws_resolved):
            raise HTTPException(status_code=403, detail="Path traversal blocked")

        # Whitelist check
        try:
            rel = target.relative_to(workspace.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Path traversal blocked")
        parts = rel.parts
        allowed = False
        if len(parts) == 1 and parts[0] in _FILE_WHITELIST:
            allowed = True
        elif len(parts) == 2 and parts[0] == "memory" and parts[1].endswith(".md"):
            allowed = True
        elif len(parts) == 3 and parts[0] == "memory" and parts[1] == "sessions" and parts[2].endswith(".md"):
            allowed = True

        if not allowed:
            raise HTTPException(status_code=403, detail="File not in whitelist")

        if not target.exists():
            raise HTTPException(status_code=404, detail="File not found")

        content = target.read_text(encoding="utf-8", errors="replace")
        return {"filename": str(rel), "content": content}

    @app.get("/v1/agents/{name}/files")
    async def list_agent_files(name: str):
        """List available workspace files for viewing."""
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        if name not in agents_cfg:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

        workspace = Path(agents_cfg[name].get("workspace", f"workspaces/{name}"))
        if not workspace.is_absolute():
            workspace = config.PROJECT_ROOT / workspace

        files = []
        # Workspace root files
        for f in _FILE_WHITELIST:
            if (workspace / f).exists():
                files.append(f)
        # Daily logs
        memory_dir = workspace / "memory"
        if memory_dir.exists():
            for f in sorted(memory_dir.glob("*.md"), reverse=True)[:7]:
                files.append(f"memory/{f.name}")
        # Session snapshots
        sessions_dir = memory_dir / "sessions"
        if sessions_dir.exists():
            for f in sorted(sessions_dir.glob("*.md"), reverse=True)[:5]:
                files.append(f"memory/sessions/{f.name}")

        return {"agent": name, "files": files}

    @app.post("/v1/admin/reload")
    async def reload_daemon():
        """
        Restart the daemon process to pick up config changes.

        Spawns a new daemon process, then schedules SIGTERM of the
        current process so the HTTP response is sent before shutdown.
        The new daemon waits for the port to be free before binding.
        """
        import signal
        import subprocess

        pid_file = config.DATA_DIR / "discord.pid"

        if not pid_file.exists():
            return {"status": "no_daemon", "message": "No daemon PID file found. Start with: alfred start"}

        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
        except (ValueError, ProcessLookupError, PermissionError):
            pid_file.unlink(missing_ok=True)
            return {"status": "stale_pid", "message": "Daemon PID is stale. Restart manually: alfred start"}

        try:
            alfred_script = config.PROJECT_ROOT / "alfred"
            log_file = config.DATA_DIR / "alfred.log"

            # Spawn the new daemon — it will wait for the port to be free
            lf = open(log_file, "a")
            subprocess.Popen(
                [str(alfred_script), "start", "--daemon-child"],
                stdout=lf,
                stderr=lf,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            lf.close()
            logger.info("New daemon process spawned for restart")

            # Schedule self-termination after a short delay so the
            # HTTP response is sent back to the client first.
            # SIGTERM lets discord.py attempt a graceful close;
            # a daemon thread escalates to SIGKILL after 5s to
            # prevent a zombie that holds the port.
            import threading as _threading

            def _force_kill_fallback():
                """Daemon thread — runs even after SIGTERM tears down asyncio."""
                import time as _time
                _time.sleep(6)  # 0.5s delay + 5s grace
                try:
                    os.kill(pid, 0)
                    logger.warning(f"PID {pid} still alive after SIGTERM, sending SIGKILL")
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass  # already dead

            _threading.Thread(
                target=_force_kill_fallback, daemon=True, name="reload-kill"
            ).start()

            async def _delayed_shutdown():
                await asyncio.sleep(0.5)
                logger.info(f"Sending SIGTERM to self (PID {pid}) for restart")
                os.kill(pid, signal.SIGTERM)

            asyncio.ensure_future(_delayed_shutdown())

            return {"status": "restarting", "message": "Restart initiated — new daemon spawning"}

        except Exception as e:
            logger.error(f"Reload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ─── Static Files (mount last so it doesn't shadow API routes) ───

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
