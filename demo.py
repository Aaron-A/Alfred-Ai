#!/usr/bin/env python3
"""
Alfred AI - Demo
Demonstrates the memory-first agent framework:
1. Store typed trade memories
2. Semantic search across them
3. Metadata filtering
4. Cross-type search
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from core.memory import MemoryStore
from core.embeddings import get_embedding_engine
from models.trade import TradeMemory, MacroMemory
from models.social import TweetMemory, DecisionMemory


def main():
    print("=" * 60)
    print("  ALFRED AI - Memory Layer Demo")
    print("=" * 60)
    print()

    # Initialize memory store
    print("[1] Initializing memory store...")
    store = MemoryStore(agent_id="trader")
    print()

    # =========================================================
    # Store some trade memories
    # =========================================================
    print("[2] Storing trade memories...")

    trades = [
        TradeMemory(
            symbol="TSLA",
            side="long",
            strategy="ORB",
            entry_price=248.50,
            exit_price=255.20,
            quantity=50,
            stop_price=245.00,
            target_price=256.00,
            pnl=335.00,
            pnl_percent=2.7,
            outcome="win",
            risk_reward_actual=1.9,
            reasoning="Strong opening range breakout above $248 resistance with 2x average volume. 9 EMA above 21 EMA on 15-min. Daily trend bullish above 50 SMA.",
            lessons="ORB works best on TSLA when pre-market volume is above 5M shares. The breakout was clean — no wicks above resistance before the move.",
            macro_context="Post-FOMC rally day. Fed held rates. VIX dropped from 18 to 15. Risk-on environment.",
            indicators="RSI(14): 62, MACD: bullish crossover, Volume: 2.1x avg, ATR: 5.80",
            entry_time="2026-02-14 09:46:00",
            exit_time="2026-02-14 11:30:00",
            hold_duration_minutes=104,
            content="TSLA ORB trade on Fed day rally. Clean breakout with volume confirmation.",
            agent_id="trader",
            tags="TSLA,ORB,win,FOMC",
        ),
        TradeMemory(
            symbol="TSLA",
            side="long",
            strategy="ORB",
            entry_price=310.00,
            exit_price=305.50,
            quantity=40,
            stop_price=306.00,
            target_price=318.00,
            pnl=-180.00,
            pnl_percent=-1.45,
            outcome="loss",
            risk_reward_actual=-1.1,
            reasoning="Attempted ORB long on TSLA after gap up. Broke above 15-min high but volume was only 1.2x average — below the 1.5x threshold.",
            lessons="Violated the volume rule. Setup looked good visually but volume was weak. Should have waited for volume confirmation. Also, CPI data was due tomorrow which added uncertainty.",
            macro_context="Day before CPI release. Markets choppy and uncertain. VIX at 22 — elevated.",
            indicators="RSI(14): 58, MACD: flat, Volume: 1.2x avg (BELOW THRESHOLD), ATR: 7.20",
            entry_time="2026-02-10 09:47:00",
            exit_time="2026-02-10 10:15:00",
            hold_duration_minutes=28,
            content="TSLA ORB loss. Violated volume confirmation rule. Pre-CPI uncertainty.",
            agent_id="trader",
            tags="TSLA,ORB,loss,volume_violation",
        ),
        TradeMemory(
            symbol="BTC/USD",
            side="long",
            strategy="BB_RSI",
            entry_price=96500.00,
            exit_price=98200.00,
            quantity=1,
            stop_price=95200.00,
            target_price=99000.00,
            pnl=1700.00,
            pnl_percent=1.76,
            outcome="win",
            risk_reward_actual=1.3,
            reasoning="BTC touched lower Bollinger Band on 15-min chart with RSI at 28. Price above 50 EMA on 1H chart confirming uptrend. Clean mean reversion setup.",
            lessons="BB+RSI works well on BTC during low-volatility weekends. The bounce was textbook — RSI divergence at the lower band.",
            macro_context="Weekend trading. No major events. BTC dominance rising. Altcoins weak. Flight to quality within crypto.",
            indicators="RSI(14): 28, BB lower touch, 50 EMA 1H: 95800, ATR: 1850",
            entry_time="2026-02-15 14:30:00",
            exit_time="2026-02-15 22:15:00",
            hold_duration_minutes=465,
            content="BTC Bollinger Band bounce. RSI oversold at lower band with uptrend intact on 1H.",
            agent_id="trader",
            tags="BTC,BB_RSI,win,weekend",
        ),
        TradeMemory(
            symbol="NVDA",
            side="long",
            strategy="TEMA_crossover",
            entry_price=142.30,
            exit_price=139.80,
            quantity=100,
            stop_price=139.50,
            target_price=148.00,
            pnl=-250.00,
            pnl_percent=-1.76,
            outcome="loss",
            risk_reward_actual=-0.44,
            reasoning="TEMA(5) crossed above TEMA(10) on 15-min with ADX at 25. RSI at 55 — in the acceptable range. However, NVDA was approaching earnings and IV was elevated.",
            lessons="Avoid TEMA crossover trades within 5 days of earnings. Implied volatility crush risk makes the R:R unfavorable even if the technical setup is valid.",
            macro_context="3 days before NVDA earnings. Market rotating out of semis into defensives. VIX at 19.",
            indicators="RSI(14): 55, ADX: 25, TEMA(5) > TEMA(10), IV rank: 85%",
            entry_time="2026-02-12 10:30:00",
            exit_time="2026-02-12 14:45:00",
            hold_duration_minutes=255,
            content="NVDA TEMA crossover loss near earnings. IV risk not properly accounted for.",
            agent_id="trader",
            tags="NVDA,TEMA,loss,earnings_risk",
        ),
        TradeMemory(
            symbol="GOOGL",
            side="long",
            strategy="VWAP_pullback",
            entry_price=185.20,
            exit_price=186.90,
            quantity=80,
            stop_price=183.80,
            target_price=187.50,
            pnl=136.00,
            pnl_percent=0.92,
            outcome="win",
            risk_reward_actual=1.2,
            reasoning="GOOGL pulled back 0.4% below VWAP with RSI(2) at 8 on 15-min chart. Daily trend bullish above 50 SMA. Clean pullback in an uptrending stock.",
            lessons="VWAP pullback on GOOGL is most reliable between 10:30 AM - 1:00 PM. Morning volatility makes early entries risky. Partial exits at VWAP reversion work well.",
            macro_context="Normal trading day. No major events. Tech sector slightly positive. GOOGL had positive analyst upgrade earlier in the week.",
            indicators="RSI(2): 8, VWAP deviation: -0.4%, Daily SMA(50): 182.50, Volume: normal",
            entry_time="2026-02-13 11:15:00",
            exit_time="2026-02-13 13:00:00",
            hold_duration_minutes=105,
            content="GOOGL VWAP pullback trade. Textbook entry on RSI(2) oversold below VWAP.",
            agent_id="trader",
            tags="GOOGL,VWAP,win,pullback",
        ),
    ]

    ids = store.store_many(trades)
    print(f"   Stored {len(ids)} trade memories")
    print()

    # =========================================================
    # Store macro event memories
    # =========================================================
    print("[3] Storing macro event memories...")

    macros = [
        MacroMemory(
            event_type="FOMC",
            event_date="2026-02-14",
            impact="high",
            market_reaction="S&P rallied 1.2% after Fed held rates. Growth stocks outperformed. VIX crushed from 18 to 15.",
            trading_notes="Post-FOMC days are excellent for ORB trades — volatility expansion from the squeeze. Reduce position sizing on FOMC day itself.",
            vix_level=15.0,
            spy_change_percent=1.2,
            btc_change_percent=3.5,
            content="FOMC rate hold decision. Markets rallied on dovish guidance.",
            agent_id="trader",
            tags="FOMC,rates,dovish,rally",
        ),
        MacroMemory(
            event_type="CPI",
            event_date="2026-02-11",
            impact="high",
            market_reaction="CPI came in hotter than expected at 3.2% vs 3.0% estimate. Markets sold off 0.8% initially then recovered by close. Choppy price action.",
            trading_notes="Hot CPI days create whipsaw — avoid ORB trades. Better to wait for dust to settle and trade VWAP pullbacks in the afternoon.",
            vix_level=22.0,
            spy_change_percent=-0.3,
            btc_change_percent=-2.1,
            content="Hot CPI print caused initial selloff then recovery. Choppy trading day.",
            agent_id="trader",
            tags="CPI,inflation,hot,choppy",
        ),
    ]

    macro_ids = store.store_many(macros)
    print(f"   Stored {len(macro_ids)} macro memories")
    print()

    # =========================================================
    # Store tweet memories
    # =========================================================
    print("[4] Storing tweet memories...")

    tweets = [
        TweetMemory(
            tweet_text="TSLA ORB breakout this morning was textbook. Volume 2x avg, clean break above the 15-min range. This is the setup we live for.",
            topic="TSLA",
            sentiment="bullish",
            likes=47,
            retweets=12,
            replies=8,
            impressions=2300,
            engagement_rate=2.9,
            posting_strategy="Share real-time trade commentary to build credibility. Show work, not just results.",
            content="TSLA ORB trade commentary tweet with high engagement.",
            agent_id="x-social",
            tags="TSLA,ORB,commentary,high_engagement",
        ),
    ]

    tweet_ids = store.store_many(tweets)
    print(f"   Stored {len(tweet_ids)} tweet memories")
    print()

    # =========================================================
    # SEMANTIC SEARCH DEMOS
    # =========================================================
    print("=" * 60)
    print("  SEARCH DEMOS")
    print("=" * 60)
    print()

    # Search 1: What happened when we traded around FOMC?
    print("[Search 1] 'What happened when we traded TSLA around the FOMC meeting?'")
    print("-" * 60)
    results = store.search("TSLA trading around FOMC rate decision", top_k=3)
    for r in results:
        dist = r.get("_distance", 0)
        print(f"  [{r.get('memory_type')}] {r.get('symbol', r.get('event_type', ''))}: {r.get('content', '')[:80]}")
        print(f"    Relevance: {max(0, 1-dist):.0%} | Outcome: {r.get('outcome', r.get('market_reaction', '')[:50])}")
        print()

    # Search 2: When did volume confirmation fail?
    print("[Search 2] 'trades where volume was too low or confirmation failed'")
    print("-" * 60)
    results = store.search("weak volume no confirmation failed volume threshold", memory_type="trade", top_k=3)
    for r in results:
        dist = r.get("_distance", 0)
        print(f"  {r.get('symbol')} {r.get('strategy')}: {r.get('lessons', '')[:100]}")
        print(f"    Relevance: {max(0, 1-dist):.0%} | PnL: ${r.get('pnl', 0):+.2f}")
        print()

    # Search 3: Metadata filter — only winning trades
    print("[Search 3] 'best setups' filtered to outcome = 'win'")
    print("-" * 60)
    results = store.search("best performing trading setups", memory_type="trade", where="outcome = 'win'", top_k=3)
    for r in results:
        dist = r.get("_distance", 0)
        print(f"  {r.get('symbol')} {r.get('strategy')}: ${r.get('pnl', 0):+.2f} ({r.get('pnl_percent', 0):+.1f}%)")
        print(f"    Reasoning: {r.get('reasoning', '')[:100]}")
        print()

    # Search 4: Cross-type search — what do we know about CPI?
    print("[Search 4] 'CPI impact on trading' (cross-type: trades + macro)")
    print("-" * 60)
    results = store.search("CPI inflation data impact on trading decisions", top_k=5)
    for r in results:
        dist = r.get("_distance", 0)
        mtype = r.get("memory_type")
        print(f"  [{mtype}] {r.get('content', '')[:90]}")
        print(f"    Relevance: {max(0, 1-dist):.0%}")
        print()

    # Search 5: What lessons have we learned?
    print("[Search 5] 'lessons learned from losing trades'")
    print("-" * 60)
    results = store.search("mistakes lessons learned from bad trades what went wrong", memory_type="trade", where="outcome = 'loss'", top_k=3)
    for r in results:
        dist = r.get("_distance", 0)
        print(f"  {r.get('symbol')} {r.get('strategy')}: {r.get('lessons', '')[:120]}")
        print(f"    Relevance: {max(0, 1-dist):.0%}")
        print()

    # =========================================================
    # STATS
    # =========================================================
    print("=" * 60)
    print("  MEMORY STATS")
    print("=" * 60)
    print()
    stats = store.stats()
    for table_name, count in stats.items():
        print(f"  {table_name}: {count} records")
    print()
    print("Done! Alfred's memory layer is operational.")


if __name__ == "__main__":
    main()
