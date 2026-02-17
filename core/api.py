"""
Alfred AI - HTTP API Server
FastAPI endpoints for interacting with Alfred agents remotely.

Start with: alfred api start
Or:         alfred api start --port 8080

Endpoints:
    POST /v1/chat              Send a message to an agent, get response
    POST /v1/memory/search     Search agent memory
    POST /v1/memory/store      Store a new memory
    GET  /v1/agents            List all agents
    GET  /v1/agents/{name}     Get agent details + session info
    POST /v1/agents/{name}/reset  Reset an agent's session
    GET  /v1/status            System status (providers, services, etc.)
    GET  /health               Health check
"""

import os
import time
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import config, _load_config, CONFIG_FILE
from .agent import Agent, AgentConfig, AgentManager


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

        # Check Discord
        discord_cfg = cfg.get("discord", {})
        discord_running = False
        if discord_cfg.get("bot_token"):
            from .discord import is_bot_running
            discord_running = is_bot_running() is not None

        return {
            "version": cfg.get("version", "0.1.0"),
            "primary_llm": primary or None,
            "secondary_llm": secondary or None,
            "providers": list(cfg.get("providers", {}).keys()),
            "agents": list(cfg.get("agents", {}).keys()),
            "discord": {
                "configured": bool(discord_cfg.get("bot_token")),
                "running": discord_running,
                "channels": len(discord_cfg.get("channels", {})),
            },
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

        return ChatResponse(
            agent=req.agent,
            response=response,
            session_id=session_id,
            turns=agent.session_info["turns"],
            elapsed_ms=elapsed,
        )

    # ─── Agents ──────────────────────────────────────────

    @app.get("/v1/agents")
    async def list_agents():
        cfg = _load_config()
        agents = cfg.get("agents", {})
        result = []
        for name, acfg in agents.items():
            result.append({
                "name": name,
                "description": acfg.get("description", ""),
                "status": acfg.get("status", "active"),
                "workspace": acfg.get("workspace", ""),
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

    # ─── Memory ──────────────────────────────────────────

    @app.post("/v1/memory/search")
    async def memory_search(req: MemorySearchRequest):
        from .memory import MemoryStore

        loop = asyncio.get_event_loop()

        def _search():
            store = MemoryStore(agent_id=req.agent_id or "default")
            return store.search(
                query=req.query,
                memory_type=req.memory_type,
                top_k=req.top_k,
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
            store = MemoryStore(agent_id=req.agent_id or "default")
            record = MemoryRecord(
                content=req.content,
                memory_type=req.memory_type,
                tags=req.tags,
            )
            return store.store(record)

        try:
            record_id = await loop.run_in_executor(None, _store)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return MemoryResponse(id=record_id, message="Memory stored")

    return app
