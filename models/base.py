"""
Alfred AI - Base Memory Model
All memory types inherit from this base.
"""

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class MemoryRecord:
    """Base memory record. All typed memories extend this."""

    # Core fields
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: str = ""  # The main text content that gets embedded
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = ""  # Where this memory came from (agent, user, system)
    agent_id: str = ""  # Which agent created this
    tags: str = ""  # Comma-separated tags for filtering
    memory_type: str = "generic"  # Discriminator field
    importance: float = 0.5  # 0.0-1.0 — how critical this memory is (affects search ranking)
    linked_ids: str = ""  # Comma-separated IDs for outcome linking (e.g., decision → trade)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return asdict(self)

    def to_embed_text(self) -> str:
        """Text to embed for vector search.

        Override in subclasses to include structured fields
        in the embedding for better semantic search.
        """
        return self.content
