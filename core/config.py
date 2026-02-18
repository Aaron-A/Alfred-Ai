"""
Alfred AI - Configuration
Central config loaded from alfred.json + environment variables.

LLM Resolution Order (per agent):
  1. Agent-specific provider/model (if set in agent config)
  2. Global primary provider/model
  3. Global secondary provider/model (fallback)
"""

import os
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.resolve()
load_dotenv(_project_root / ".env")

# Config file path
CONFIG_FILE = _project_root / "alfred.json"


def _load_config() -> dict:
    """Load config from alfred.json if it exists."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def _save_config(data: dict):
    """Save config to alfred.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


class Config:
    """Central configuration for Alfred AI."""

    # Paths
    PROJECT_ROOT: Path = _project_root
    DATA_DIR: Path = _project_root / "data"
    LANCEDB_DIR: Path = _project_root / "data" / "lancedb"
    MODELS_CACHE_DIR: Path = _project_root / "data" / "models"

    # Embedding model
    EMBEDDING_MODEL: str = os.getenv("ALFRED_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    EMBEDDING_DIMENSION: int = int(os.getenv("ALFRED_EMBEDDING_DIM", "768"))
    EMBEDDING_PREFIX_SEARCH: str = "search_query: "
    EMBEDDING_PREFIX_DOCUMENT: str = "search_document: "

    # Memory
    MEMORY_TABLE_PREFIX: str = "memory_"
    CHUNK_SIZE: int = int(os.getenv("ALFRED_CHUNK_SIZE", "400"))
    CHUNK_OVERLAP: int = int(os.getenv("ALFRED_CHUNK_OVERLAP", "80"))

    # Search
    DEFAULT_TOP_K: int = int(os.getenv("ALFRED_TOP_K", "10"))
    HYBRID_VECTOR_WEIGHT: float = float(os.getenv("ALFRED_VECTOR_WEIGHT", "0.7"))
    HYBRID_TEXT_WEIGHT: float = float(os.getenv("ALFRED_TEXT_WEIGHT", "0.3"))

    # ─── Timezone ─────────────────────────────────────────────────

    @property
    def TIMEZONE(self) -> str:
        """Configured timezone name (IANA format). Reads from alfred.json settings.timezone."""
        cfg = _load_config()
        return cfg.get("settings", {}).get("timezone", "UTC")

    @property
    def tz(self) -> ZoneInfo:
        """Get ZoneInfo object for the configured timezone."""
        return ZoneInfo(self.TIMEZONE)

    # ─── LLM Provider Config ────────────────────────────────────

    # Default models per provider
    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-6",
        "xai": "grok-4-1-fast-reasoning",
        "openai": "gpt-5.2",
        "ollama": "llama3.1",
    }

    # ─── Primary / Secondary ────────────────────────────────────

    @property
    def LLM_PROVIDER(self) -> str:
        """Get the primary LLM provider."""
        cfg = _load_config()
        llm = cfg.get("llm", {})
        return llm.get("primary", {}).get("provider", llm.get("provider", "anthropic"))

    @property
    def LLM_MODEL(self) -> str:
        """Get the primary LLM model."""
        cfg = _load_config()
        llm = cfg.get("llm", {})
        return llm.get("primary", {}).get("model", llm.get("model", "claude-sonnet-4-5-20250929"))

    @property
    def LLM_SECONDARY_PROVIDER(self) -> str:
        """Get the secondary (fallback) LLM provider."""
        cfg = _load_config()
        return cfg.get("llm", {}).get("secondary", {}).get("provider", "")

    @property
    def LLM_SECONDARY_MODEL(self) -> str:
        """Get the secondary (fallback) LLM model."""
        cfg = _load_config()
        return cfg.get("llm", {}).get("secondary", {}).get("model", "")

    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider. Checks alfred.json then env vars."""
        cfg = _load_config()

        # Check alfred.json first
        providers = cfg.get("providers", {})
        if provider in providers and "api_key" in providers[provider]:
            return providers[provider]["api_key"]

        # Fall back to environment variables
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "xai": "XAI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "brave": "BRAVE_API_KEY",
            "ollama": "",
        }
        env_var = env_map.get(provider, "")
        return os.getenv(env_var, "") if env_var else ""

    def get_default_model(self, provider: str) -> str:
        """Get the default model for a provider."""
        cfg = _load_config()
        providers = cfg.get("providers", {})
        if provider in providers and "model" in providers[provider]:
            return providers[provider]["model"]
        return self.DEFAULT_MODELS.get(provider, "")

    def get_service_credentials(self, service: str) -> dict:
        """
        Get credentials for an external service (non-LLM APIs).

        Resolution order:
          1. alfred.json services.<name> section
          2. Environment variables: <SERVICE>_API_KEY, <SERVICE>_SECRET_KEY, etc.

        Returns dict with whatever keys are configured (api_key, secret_key,
        base_url, data_url, etc.), or empty dict if not found.
        """
        cfg = _load_config()

        # Check alfred.json first
        services = cfg.get("services", {})
        if service in services:
            return dict(services[service])

        # Fall back to environment variables
        prefix = service.upper()
        result = {}
        env_mappings = {
            "api_key": f"{prefix}_API_KEY",
            "secret_key": f"{prefix}_SECRET_KEY",
            "base_url": f"{prefix}_BASE_URL",
            "data_url": f"{prefix}_DATA_URL",
        }
        for key, env_var in env_mappings.items():
            val = os.getenv(env_var, "")
            if val:
                result[key] = val

        return result

    def get_service_domains(self) -> dict[str, dict]:
        """
        Build a domain→service mapping for auto-injection.

        Returns dict like:
          {"paper-api.alpaca.markets": {"service": "alpaca", ...creds...},
           "data.alpaca.markets": {"service": "alpaca", ...creds...}}
        """
        cfg = _load_config()
        services = cfg.get("services", {})
        domain_map = {}

        for name, svc_cfg in services.items():
            creds = self.get_service_credentials(name)
            # Extract domains from any URL fields
            for key, val in svc_cfg.items():
                if key.endswith("_url") or key == "base_url":
                    try:
                        from urllib.parse import urlparse
                        host = urlparse(val).hostname
                        if host:
                            domain_map[host] = {"service": name, **creds}
                    except Exception:
                        pass

        return domain_map

    def get_agent_llm(self, agent_name: str) -> tuple[str, str]:
        """
        Get the provider and model for a specific agent.

        Resolution order:
          1. Agent-specific override
          2. Global primary
          3. Global secondary
        """
        cfg = _load_config()
        agent_cfg = cfg.get("agents", {}).get(agent_name, {})

        # Agent-level override
        if agent_cfg.get("provider"):
            return agent_cfg["provider"], agent_cfg.get("model", self.get_default_model(agent_cfg["provider"]))

        # Global primary
        return self.LLM_PROVIDER, self.LLM_MODEL

    @classmethod
    def ensure_dirs(cls):
        """Create required directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_configured(self) -> bool:
        """Check if Alfred has been set up (alfred.json exists)."""
        return CONFIG_FILE.exists()


config = Config()
