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
        allow_credentials=True,
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
    async def get_metrics(agent: str = None, period: str = "session"):
        """
        Get agent activity metrics.

        Args:
            agent: Filter to a specific agent
            period: session (current in-memory), day, week, month, year, all
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

    # Curated provider → models registry.
    # Update these lists as new providers/models become available.
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
            "llama3.1",
            "llama3.2",
            "mistral",
            "codellama",
            "deepseek-r1",
            "qwen2.5",
        ],
    }

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

    # ─── Agent Config Update ──────────────────────────────

    class AgentConfigUpdate(BaseModel):
        provider: Optional[str] = None
        model: Optional[str] = None

    @app.patch("/v1/agents/{name}/config")
    async def update_agent_config(name: str, update: AgentConfigUpdate):
        """
        Update an agent's provider/model in alfred.json.

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

        cfg["agents"] = agents_cfg
        _save_config(cfg)

        logger.info(f"Updated agent '{name}' config: provider={update.provider}, model={update.model}")

        return {
            "message": f"Agent '{name}' config updated",
            "provider": agents_cfg[name].get("provider"),
            "model": agents_cfg[name].get("model"),
            "restart_required": True,
        }

    @app.post("/v1/admin/reload")
    async def reload_daemon():
        """
        Restart the daemon process to pick up config changes.

        Sends SIGHUP to the daemon process if running, which triggers
        a graceful restart. If no daemon is running, returns a message
        telling the user to start it manually.
        """
        import signal

        pid_file = config.DATA_DIR / "daemon.pid"

        if not pid_file.exists():
            return {"status": "no_daemon", "message": "No daemon PID file found. Start with: alfred daemon start"}

        try:
            pid = int(pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            # PID file is stale
            pid_file.unlink(missing_ok=True)
            return {"status": "stale_pid", "message": "Daemon PID is stale. Restart manually: alfred daemon start"}

        try:
            # Send SIGTERM to trigger graceful shutdown, then caller can restart
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to daemon PID {pid}")

            # Wait briefly for process to die
            for _ in range(10):
                await asyncio.sleep(0.3)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pid_file.unlink(missing_ok=True)
                    # Now restart the daemon
                    import subprocess
                    subprocess.Popen(
                        ["python", "-m", "cli.main", "daemon", "start"],
                        cwd=str(config.PROJECT_ROOT),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    logger.info("Daemon restart initiated")
                    return {"status": "restarted", "message": "Daemon restarted successfully"}

            return {"status": "timeout", "message": f"Daemon PID {pid} did not stop in time. Kill manually."}

        except Exception as e:
            logger.error(f"Reload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ─── Static Files (mount last so it doesn't shadow API routes) ───

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
