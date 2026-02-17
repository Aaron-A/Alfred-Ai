"""
Alfred AI - Social Memory Model
Structured memory for tweets and social media engagement.
"""

from dataclasses import dataclass, field
from .base import MemoryRecord


@dataclass
class TweetMemory(MemoryRecord):
    """A tweet posted or engaged with."""

    memory_type: str = "tweet"

    # Tweet details
    tweet_text: str = ""
    tweet_url: str = ""
    tweet_id: str = ""
    author: str = ""
    topic: str = ""  # e.g., "TSLA", "BTC", "AI", "macro"
    sentiment: str = ""  # "bullish", "bearish", "neutral", "hype"

    # Engagement (updated after posting)
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0

    # Context
    is_reply: bool = False
    reply_to_url: str = ""
    posting_strategy: str = ""  # Why this was posted / what approach

    def to_embed_text(self) -> str:
        parts = [
            f"Tweet about {self.topic}" if self.topic else "Tweet",
            f"by {self.author}" if self.author else "",
            f"Sentiment: {self.sentiment}" if self.sentiment else "",
            f'Text: "{self.tweet_text}"' if self.tweet_text else "",
            f"Strategy: {self.posting_strategy}" if self.posting_strategy else "",
            f"Engagement: {self.likes} likes, {self.retweets} RTs" if self.likes else "",
            self.content,
        ]
        return " | ".join(p for p in parts if p)


@dataclass
class DecisionMemory(MemoryRecord):
    """A decision made by any agent, with context and outcome tracking."""

    memory_type: str = "decision"

    # Decision details
    decision: str = ""  # What was decided
    context: str = ""  # Why this decision was faced
    options_considered: str = ""  # What alternatives were weighed
    rationale: str = ""  # Why this option was chosen

    # Outcome (filled in later)
    outcome: str = ""  # What happened
    outcome_quality: str = ""  # "good", "bad", "neutral", "unknown"
    regret_score: int = 0  # 0-10, how much we'd change this in hindsight

    def to_embed_text(self) -> str:
        parts = [
            f"Decision: {self.decision}" if self.decision else "",
            f"Context: {self.context}" if self.context else "",
            f"Rationale: {self.rationale}" if self.rationale else "",
            f"Outcome: {self.outcome} ({self.outcome_quality})" if self.outcome else "",
            self.content,
        ]
        return " | ".join(p for p in parts if p)
