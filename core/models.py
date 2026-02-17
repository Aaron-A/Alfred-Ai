"""
Alfred AI - Model Discovery
Fetch available models from provider APIs.

Usage:
    alfred models update          # Fetch latest from all configured providers
    alfred models list            # Show cached model lists
    alfred models list anthropic  # Show models for one provider
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from .config import config, _load_config


# Cache file for discovered models
MODELS_CACHE = config.DATA_DIR / "models_cache.json"


def _load_cache() -> dict:
    """Load the models cache file."""
    if MODELS_CACHE.exists():
        with open(MODELS_CACHE) as f:
            return json.load(f)
    return {}


def _save_cache(data: dict):
    """Save the models cache file."""
    MODELS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_CACHE, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def fetch_anthropic_models(api_key: str) -> list[dict]:
    """Fetch available models from Anthropic API."""
    url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "User-Agent": "Alfred-AI/1.0",
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = []
            for m in data.get("data", []):
                models.append({
                    "id": m.get("id", ""),
                    "name": m.get("display_name", m.get("id", "")),
                    "created": m.get("created_at", ""),
                })
            # Sort by created date descending (newest first)
            models.sort(key=lambda x: x.get("created", ""), reverse=True)
            return models
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API error ({e.code}): {body[:200]}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Anthropic models: {e}")


def fetch_xai_models(api_key: str) -> list[dict]:
    """Fetch available models from xAI API."""
    url = "https://api.x.ai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Alfred-AI/1.0",
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = []
            for m in data.get("data", []):
                models.append({
                    "id": m.get("id", ""),
                    "name": m.get("id", ""),
                    "created": m.get("created", 0),
                })
            models.sort(key=lambda x: x.get("created", 0), reverse=True)
            return models
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"xAI API error ({e.code}): {body[:200]}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch xAI models: {e}")


def fetch_openai_models(api_key: str) -> list[dict]:
    """Fetch available models from OpenAI API."""
    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Alfred-AI/1.0",
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                # Filter to chat models only (skip embeddings, tts, whisper, etc.)
                if any(prefix in model_id for prefix in ["gpt-", "o1", "o3", "o4", "chatgpt"]):
                    models.append({
                        "id": model_id,
                        "name": model_id,
                        "created": m.get("created", 0),
                    })
            models.sort(key=lambda x: x.get("created", 0), reverse=True)
            return models
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error ({e.code}): {body[:200]}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OpenAI models: {e}")


def fetch_ollama_models() -> list[dict]:
    """Fetch models from local Ollama instance."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"User-Agent": "Alfred-AI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = []
            for m in data.get("models", []):
                models.append({
                    "id": m.get("name", ""),
                    "name": m.get("name", ""),
                    "size": m.get("size", 0),
                })
            return models
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable: {e}")


# Provider -> fetch function mapping
FETCHERS = {
    "anthropic": fetch_anthropic_models,
    "xai": fetch_xai_models,
    "openai": fetch_openai_models,
    "ollama": fetch_ollama_models,
}


def update_models(provider_id: str = None) -> dict:
    """
    Fetch latest models from provider APIs and update the cache.

    Args:
        provider_id: Specific provider, or None for all configured providers.

    Returns:
        Dict of {provider: [model_list]} that was fetched.
    """
    cfg = _load_config()
    providers = cfg.get("providers", {})
    cache = _load_cache()

    results = {}

    targets = [provider_id] if provider_id else list(providers.keys())

    for pid in targets:
        if pid not in FETCHERS:
            continue

        pcfg = providers.get(pid, {})
        api_key = pcfg.get("api_key", "")

        fetcher = FETCHERS[pid]

        try:
            if pid == "ollama":
                models = fetcher()
            else:
                if not api_key:
                    raise RuntimeError(f"No API key configured for {pid}")
                models = fetcher(api_key)

            results[pid] = models
            cache[pid] = {
                "models": models,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        except Exception as e:
            results[pid] = {"error": str(e)}

    _save_cache(cache)
    return results


def get_cached_models(provider_id: str = None) -> dict:
    """Get cached model lists."""
    cache = _load_cache()
    if provider_id:
        return {provider_id: cache.get(provider_id, {})}
    return cache
