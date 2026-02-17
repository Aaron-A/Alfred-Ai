"""
Alfred AI - Trade Memory Model
Structured memory for trading decisions and outcomes.
"""

from dataclasses import dataclass, field
from typing import Optional
from .base import MemoryRecord


@dataclass
class TradeMemory(MemoryRecord):
    """A trade entry/exit with full context."""

    memory_type: str = "trade"

    # Trade details
    symbol: str = ""
    side: str = ""  # "long" or "short"
    strategy: str = ""  # e.g., "ORB", "VWAP_pullback", "BB_RSI"
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    stop_price: float = 0.0
    target_price: float = 0.0

    # Outcome
    pnl: float = 0.0
    pnl_percent: float = 0.0
    outcome: str = ""  # "win", "loss", "breakeven", "open"
    risk_reward_actual: float = 0.0

    # Context
    reasoning: str = ""  # Why the trade was taken
    lessons: str = ""  # What was learned
    macro_context: str = ""  # Market conditions at time of trade
    indicators: str = ""  # Key indicator readings at entry

    # Timing
    entry_time: str = ""
    exit_time: str = ""
    hold_duration_minutes: int = 0

    def to_embed_text(self) -> str:
        """Rich text for embedding — includes reasoning, context, lessons."""
        parts = [
            f"{self.side} {self.symbol} via {self.strategy} strategy",
            f"Reasoning: {self.reasoning}" if self.reasoning else "",
            f"Macro context: {self.macro_context}" if self.macro_context else "",
            f"Outcome: {self.outcome}, PnL: ${self.pnl:.2f} ({self.pnl_percent:.1f}%)" if self.outcome else "",
            f"Lessons: {self.lessons}" if self.lessons else "",
            f"Indicators: {self.indicators}" if self.indicators else "",
            self.content,
        ]
        return " | ".join(p for p in parts if p)


@dataclass
class MacroMemory(MemoryRecord):
    """Macro economic event or context snapshot."""

    memory_type: str = "macro"

    # Event details
    event_type: str = ""  # "FOMC", "CPI", "GDP", "earnings", "geopolitical"
    event_date: str = ""
    impact: str = ""  # "high", "medium", "low"

    # Market reaction
    market_reaction: str = ""  # Description of how markets reacted
    vix_level: float = 0.0
    spy_change_percent: float = 0.0
    btc_change_percent: float = 0.0

    # Trading implications
    trading_notes: str = ""  # What this means for our strategies

    def to_embed_text(self) -> str:
        parts = [
            f"{self.event_type} event on {self.event_date}",
            f"Impact: {self.impact}" if self.impact else "",
            f"Market reaction: {self.market_reaction}" if self.market_reaction else "",
            f"Trading notes: {self.trading_notes}" if self.trading_notes else "",
            self.content,
        ]
        return " | ".join(p for p in parts if p)
