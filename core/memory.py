"""
Alfred AI - Memory Layer
LanceDB-backed vector memory with typed schemas and hybrid search.

This is the brain of Alfred. Every agent reads from and writes to this
shared memory layer. Memories are embedded locally, stored in LanceDB,
and retrievable via semantic search + metadata filtering.
"""

import math
import time
from datetime import datetime, timezone
from typing import Optional, Type
from dataclasses import asdict

import lancedb
import pyarrow as pa

from .config import config
from .embeddings import get_embedding_engine
from .logging import get_logger
from models.base import MemoryRecord

logger = get_logger("memory")


class MemoryStore:
    """
    Vector memory store backed by LanceDB.

    Supports:
    - Store typed memory records with automatic embedding
    - Semantic search (vector similarity)
    - Metadata filtering (exact match on structured fields)
    - Hybrid search (vector + BM25 full-text via LanceDB FTS index)
    - Update and delete records
    """

    def __init__(self, db_path: str = None, agent_id: str = "default"):
        config.ensure_dirs()
        self.db_path = db_path or str(config.LANCEDB_DIR)
        self.agent_id = agent_id
        self.db = lancedb.connect(self.db_path)
        self.embedder = get_embedding_engine()
        self._tables: dict[str, lancedb.table.Table] = {}
        self._fts_indexed: set[str] = set()  # Tables that have FTS indexes
        self.max_memories_per_agent = 10000  # Auto-compact when exceeded (0 = no limit)

    def _table_name(self, memory_type: str) -> str:
        """Generate table name from memory type."""
        return f"{config.MEMORY_TABLE_PREFIX}{memory_type}"

    def _count_agent_memories(self, agent_id: str) -> int:
        """Count total memories across all tables for a given agent."""
        total = 0
        for table_name in self._list_table_names():
            if not table_name.startswith(config.MEMORY_TABLE_PREFIX):
                continue
            try:
                table = self.db.open_table(table_name)
                total += len(table.to_pandas().query(f"agent_id == '{agent_id}'"))
            except Exception:
                try:
                    # Fallback: count via search
                    total += len(table.search().where(f"agent_id = '{agent_id}'").limit(100000).to_list())
                except Exception:
                    pass
        return total

    def _list_table_names(self) -> list[str]:
        """Get table names, handling both old (list) and new (response object) LanceDB APIs."""
        result = self.db.list_tables()
        # LanceDB < 0.29: returns list[str] directly
        if isinstance(result, list):
            return result
        # LanceDB >= 0.29: returns ListTablesResponse with .tables attribute
        if hasattr(result, "tables"):
            return result.tables
        # Fallback: try to coerce
        return list(result)

    def _migrate_table_schema(self, table, table_name: str):
        """
        Add missing columns to an existing LanceDB table.

        When MemoryRecord gains new fields (e.g. importance, linked_ids),
        existing tables won't have those columns. This adds them with
        defaults so writes don't fail with 'Field not found in target schema'.
        """
        schema_names = {f.name for f in table.schema}

        # Define expected columns with their SQL default expressions
        expected = {
            "importance": "0.5",
            "linked_ids": "''",
        }

        for col, default_expr in expected.items():
            if col not in schema_names:
                try:
                    table.add_columns({col: default_expr})
                    logger.info(f"Migrated {table_name}: added column '{col}'")
                except Exception as e:
                    logger.warning(f"Failed to add column '{col}' to {table_name}: {e}")

    def _get_or_create_table(self, data: list[dict]) -> lancedb.table.Table:
        """Get existing table or create one from data."""
        # Infer memory_type from the first record
        memory_type = data[0].get("memory_type", "generic")
        table_name = self._table_name(memory_type)

        if table_name in self._tables:
            return self._tables[table_name], False

        existing_tables = self._list_table_names()

        if table_name in existing_tables:
            table = self.db.open_table(table_name)
            self._migrate_table_schema(table, table_name)
            self._tables[table_name] = table
            return table, False
        else:
            # Create table with the full batch of data
            table = self.db.create_table(table_name, data=data)
            logger.info(f"Created memory table: {table_name}")
            self._tables[table_name] = table
            return table, True  # True = data was already inserted via create

    def store(self, record: MemoryRecord, dedup: bool = True) -> str:
        """
        Store a memory record with automatic embedding and optional deduplication.

        Before storing, checks if a near-identical memory exists (>= 95% similarity
        AND less than 24 hours old). If found, updates the existing record instead
        of creating a duplicate. This prevents the same hourly scan from filling
        memory with near-identical entries.

        Args:
            record: Any MemoryRecord subclass (TradeMemory, TweetMemory, etc.)
            dedup: If True, check for near-duplicates before storing (default True)

        Returns:
            The record ID (existing ID if deduped, new ID otherwise)
        """
        # Set agent_id if not already set
        if not record.agent_id:
            record.agent_id = self.agent_id

        # ─── Memory Quota Check ──────────────────────────────────
        max_memories = self.max_memories_per_agent
        if max_memories > 0:
            try:
                count = self._count_agent_memories(record.agent_id)
                if count >= max_memories:
                    logger.warning(
                        f"Memory quota reached for {record.agent_id}: "
                        f"{count}/{max_memories} — auto-compacting"
                    )
                    self.compact(max_age_days=60, min_importance=0.3)
            except Exception as e:
                logger.debug(f"Quota check failed (non-fatal): {e}")

        # Generate embedding from the rich text representation
        embed_text = record.to_embed_text()
        vector = self.embedder.embed_documents([embed_text])[0]

        # ─── Deduplication Check ─────────────────────────────────
        if dedup:
            existing_id = self._check_duplicate(record, vector)
            if existing_id:
                # Update the existing record's content and timestamp
                self.update(existing_id, record.memory_type, {
                    "content": record.content,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "tags": record.tags or "",
                })
                logger.debug(f"Dedup: updated existing memory {existing_id} instead of creating new")
                return existing_id

        # Convert to dict and add vector
        data = record.to_dict()
        data["vector"] = vector

        # Get or create table
        table, already_inserted = self._get_or_create_table([data])

        # Only add if table already existed (data not yet inserted)
        if not already_inserted:
            table.add([data])

        return record.id

    def _check_duplicate(self, record: MemoryRecord, vector: list[float]) -> Optional[str]:
        """
        Check if a near-duplicate memory exists.

        Returns the existing record's ID if found (>= 95% similar AND < 24h old),
        or None if no duplicate exists.
        """
        table_name = self._table_name(record.memory_type)

        try:
            if table_name not in self._list_table_names():
                return None

            table = self.db.open_table(table_name)

            # Search for similar records from the same agent
            where = f"agent_id = '{record.agent_id}'"
            results = table.search(vector).where(where).limit(1).to_list()

            if not results:
                return None

            top = results[0]
            distance = top.get("_distance", 1.0)
            similarity = 1.0 - distance

            # Must be >= 95% similar
            if similarity < 0.95:
                return None

            # Must be less than 24 hours old
            created_at_str = top.get("created_at", "")
            if not created_at_str:
                return None

            try:
                if created_at_str.endswith("Z"):
                    created_at_str = created_at_str[:-1] + "+00:00"
                created_at = datetime.fromisoformat(created_at_str)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                return None

            age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
            if age_seconds > 86400:  # Older than 24h — not a dup, it's historical
                return None

            return top.get("id")

        except Exception as e:
            logger.debug(f"Dedup check failed (non-fatal): {e}")
            return None

    def store_many(self, records: list[MemoryRecord]) -> list[str]:
        """Store multiple records efficiently with batch embedding."""
        if not records:
            return []

        # Group by memory_type for batch processing
        by_type: dict[str, list[MemoryRecord]] = {}
        for r in records:
            if not r.agent_id:
                r.agent_id = self.agent_id
            by_type.setdefault(r.memory_type, []).append(r)

        all_ids = []
        for memory_type, type_records in by_type.items():
            # Batch embed
            texts = [r.to_embed_text() for r in type_records]
            vectors = self.embedder.embed_documents(texts)

            # Build data dicts
            data = []
            for r, vec in zip(type_records, vectors):
                d = r.to_dict()
                d["vector"] = vec
                data.append(d)
                all_ids.append(r.id)

            # Get or create table (create_table inserts the data)
            table, already_inserted = self._get_or_create_table(data)

            # Only add if table already existed
            if not already_inserted:
                table.add(data)

        return all_ids

    def _ensure_fts_index(self, table, table_name: str):
        """
        Create a full-text search index on the 'content' column if not already done.
        FTS indexes are required for hybrid search (vector + BM25).
        """
        if table_name in self._fts_indexed:
            return True

        try:
            # Check if the table has a 'content' column (all memory records do)
            schema_names = [f.name for f in table.schema]
            if "content" not in schema_names:
                return False

            # Create FTS index — replace if it already exists (handles schema changes)
            table.create_fts_index("content", replace=True)
            self._fts_indexed.add(table_name)
            return True
        except Exception as e:
            # FTS creation can fail on empty tables or unsupported configs
            # Fall back to vector-only search silently
            return False

    def search(
        self,
        query: str,
        memory_type: str = None,
        top_k: int = None,
        where: str = None,
        filters: dict = None,
    ) -> list[dict]:
        """
        Search across memories using hybrid (vector + BM25) or vector-only search.

        If hybrid search is enabled in config and an FTS index exists,
        combines vector similarity with keyword matching for better recall.
        Falls back to vector-only search if hybrid isn't available.

        Args:
            query: Natural language search query
            memory_type: Filter to a specific memory type (e.g., "trade", "tweet")
                        If None, searches all memory tables.
            top_k: Number of results to return
            where: SQL-like filter string (e.g., "symbol = 'TSLA' AND outcome = 'win'")
            filters: Dict of field->value for structured filtering
                    (e.g., {"symbol": "BTC/USD", "outcome": "win"})

        Returns:
            List of dicts with memory fields + _relevance_score (hybrid) or _distance (vector)
        """
        import time as _time
        _search_start = _time.monotonic()

        top_k = top_k or config.DEFAULT_TOP_K

        # Merge structured filters into where clause
        if filters:
            filter_clauses = []
            for k, v in filters.items():
                if isinstance(v, str):
                    filter_clauses.append(f"{k} = '{v}'")
                elif isinstance(v, (int, float)):
                    filter_clauses.append(f"{k} = {v}")
            if filter_clauses:
                filter_where = " AND ".join(filter_clauses)
                where = f"{where} AND {filter_where}" if where else filter_where

        query_vector = self.embedder.embed_query(query)

        # Check if hybrid search is enabled in config
        use_hybrid = getattr(config, 'HYBRID_VECTOR_WEIGHT', 0) > 0

        results = []

        # Determine which tables to search
        if memory_type:
            table_names = [self._table_name(memory_type)]
        else:
            # Search all memory tables
            table_names = [
                t for t in self._list_table_names()
                if t.startswith(config.MEMORY_TABLE_PREFIX)
            ]

        for table_name in table_names:
            try:
                table = self.db.open_table(table_name)
                self._migrate_table_schema(table, table_name)

                # Try hybrid search first if enabled
                if use_hybrid and self._ensure_fts_index(table, table_name):
                    try:
                        search = (
                            table.search(query_type="hybrid")
                            .text(query)
                            .vector(query_vector)
                            .limit(top_k)
                        )
                        if where:
                            search = search.where(where)
                        table_results = search.to_list()

                        # Normalize: hybrid returns _relevance_score (higher = better)
                        # Convert to _distance equivalent for consistent downstream usage
                        for r in table_results:
                            if "_relevance_score" in r and "_distance" not in r:
                                r["_distance"] = 1.0 - min(r["_relevance_score"], 1.0)
                        results.extend(table_results)
                        continue  # Success — skip vector-only fallback
                    except Exception:
                        pass  # Fall through to vector-only

                # Vector-only search (fallback)
                search = table.search(query_vector).limit(top_k)
                if where:
                    search = search.where(where)
                table_results = search.to_list()
                results.extend(table_results)

            except Exception as e:
                logger.warning(f"Search failed on {table_name}: {e}")

        # Apply temporal decay if we have timestamps
        results = self._apply_temporal_decay(results)

        # Apply importance boost — high-importance memories rank higher
        results = self._apply_importance_boost(results)

        # Sort by combined score (lower = better) and trim to top_k
        results.sort(key=lambda x: x.get("_distance", float("inf")))
        final = results[:top_k]

        # Log vector query for observability
        _search_elapsed = int((_time.monotonic() - _search_start) * 1000)
        avg_dist = 0.0
        if final:
            avg_dist = sum(r.get("_distance", 0) for r in final) / len(final)
        self._log_vector_query(
            query=query, memory_type=memory_type, top_k=top_k,
            result_count=len(final), avg_distance=avg_dist,
            elapsed_ms=_search_elapsed,
        )

        return final

    def _log_vector_query(
        self,
        query: str,
        memory_type: Optional[str],
        top_k: int,
        result_count: int,
        avg_distance: float,
        elapsed_ms: int,
    ):
        """Log a vector search query to SQLite for observability."""
        try:
            import sqlite3
            db_path = config.DATA_DIR / "metrics.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_queries (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    agent         TEXT    DEFAULT '',
                    query_text    TEXT    NOT NULL,
                    memory_type   TEXT    DEFAULT '',
                    top_k         INTEGER DEFAULT 10,
                    result_count  INTEGER DEFAULT 0,
                    avg_distance  REAL    DEFAULT 0.0,
                    elapsed_ms    INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vq_ts ON vector_queries(timestamp)
            """)
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """INSERT INTO vector_queries
                   (timestamp, agent, query_text, memory_type, top_k, result_count, avg_distance, elapsed_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, self.agent_id, query[:500], memory_type or "", top_k,
                 result_count, round(avg_distance, 4), elapsed_ms),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass  # Never let logging break search

    def _apply_temporal_decay(
        self,
        results: list[dict],
        half_life_days: float = 30.0,
        recency_weight: float = 0.2,
    ) -> list[dict]:
        """
        Apply time-based decay to search results so recent memories rank higher.

        Blends vector similarity with a recency score using exponential decay.
        Old memories aren't penalized harshly — they just lose a small recency
        bonus. A 90-day-old memory with perfect semantic match still ranks well.

        Args:
            results: Raw search results with _distance and created_at fields
            half_life_days: Days until recency bonus drops to 50% (default 30)
            recency_weight: How much recency matters vs similarity (0.0-1.0, default 0.2)

        The math:
            similarity = 1.0 - _distance  (0 to 1, higher = more similar)
            recency = exp(-0.693 * age_days / half_life)  (1.0 for today, 0.5 at half-life)
            combined = (1 - weight) * similarity + weight * recency
            _distance = 1.0 - combined  (back to distance format, lower = better)
        """
        if not results or recency_weight <= 0:
            return results

        now = datetime.now(timezone.utc)
        decay_rate = 0.693147 / max(half_life_days, 1)  # ln(2) / half_life

        for r in results:
            # Parse created_at timestamp
            created_at_str = r.get("created_at", "")
            if not created_at_str:
                continue  # No timestamp — leave distance as-is

            try:
                # Handle both formats: with and without timezone
                if created_at_str.endswith("Z"):
                    created_at_str = created_at_str[:-1] + "+00:00"
                created_at = datetime.fromisoformat(created_at_str)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue  # Unparseable — skip

            age_days = max(0, (now - created_at).total_seconds() / 86400)

            # Exponential decay: 1.0 at age=0, 0.5 at half_life, ~0 at 5x half_life
            recency = math.exp(-decay_rate * age_days)

            # Blend similarity with recency
            raw_distance = r.get("_distance", 0.5)
            similarity = max(0, 1.0 - raw_distance)
            combined = (1.0 - recency_weight) * similarity + recency_weight * recency

            # Store back as distance (lower = better)
            r["_distance"] = 1.0 - combined
            r["_recency"] = recency
            r["_age_days"] = round(age_days, 1)

        return results

    def _apply_importance_boost(self, results: list[dict]) -> list[dict]:
        """
        Boost high-importance memories in search results.

        Memories with importance > 0.5 get a relevance bonus,
        those below 0.5 get a slight penalty. The effect is capped
        at ~20% distance adjustment so importance complements but
        never overrides semantic similarity.

        The math:
            boost = (importance - 0.5) * 0.2  (range: -0.1 to +0.1)
            adjusted_distance = distance - boost  (lower distance = better)
        """
        for r in results:
            importance = r.get("importance", 0.5)
            if importance is None:
                importance = 0.5

            # Clamp to valid range
            importance = max(0.0, min(1.0, float(importance)))

            # Calculate boost: +0.1 for importance=1.0, 0 for 0.5, -0.1 for 0.0
            boost = (importance - 0.5) * 0.2
            raw_distance = r.get("_distance", 0.5)
            r["_distance"] = max(0.0, min(1.0, raw_distance - boost))
            r["_importance"] = importance

        return results

    def compact(
        self,
        memory_type: str = None,
        max_age_days: int = 90,
        min_importance: float = 0.3,
    ) -> dict:
        """
        Prune old, low-importance memories to keep the store manageable.

        Deletes records that are BOTH:
        - Older than max_age_days
        - Below min_importance threshold

        Records with importance >= 0.7 are NEVER deleted (lessons learned are permanent).

        Args:
            memory_type: Specific type to compact (None = all tables)
            max_age_days: Only prune records older than this
            min_importance: Only prune records below this importance

        Returns:
            Dict of {table_name: records_deleted}
        """
        from datetime import timedelta

        results = {}
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cutoff_iso = cutoff.isoformat()

        # Determine which tables to compact
        if memory_type:
            table_names = [self._table_name(memory_type)]
        else:
            table_names = [
                t for t in self._list_table_names()
                if t.startswith(config.MEMORY_TABLE_PREFIX)
            ]

        for table_name in table_names:
            try:
                table = self.db.open_table(table_name)
                self._migrate_table_schema(table, table_name)

                # Find records that are old AND low importance
                # LanceDB where clause format
                where = (
                    f"created_at < '{cutoff_iso}' AND "
                    f"importance < {min_importance} AND "
                    f"importance < 0.7"  # Safety: never delete high-importance
                )

                # Count before delete
                try:
                    to_delete = table.search().where(where).limit(10000).to_list()
                    count = len(to_delete)
                except Exception:
                    count = 0

                if count > 0:
                    table.delete(where)
                    short_name = table_name.replace(config.MEMORY_TABLE_PREFIX, "")
                    logger.info(f"Compacted {short_name}: removed {count} records "
                              f"(>{max_age_days} days, importance <{min_importance})")

                results[table_name] = count

            except Exception as e:
                logger.warning(f"Compact failed on {table_name}: {e}")
                results[table_name] = 0

        return results

    def get(self, record_id: str, memory_type: str) -> Optional[dict]:
        """Get a specific record by ID."""
        table_name = self._table_name(memory_type)
        try:
            table = self.db.open_table(table_name)
            results = table.search().where(f"id = '{record_id}'").limit(1).to_list()
            return results[0] if results else None
        except Exception:
            return None

    def update(self, record_id: str, memory_type: str, updates: dict) -> bool:
        """
        Update fields on an existing record.

        Args:
            record_id: The record ID to update
            memory_type: The memory type table
            updates: Dict of field -> new value

        Returns:
            True if updated, False if not found
        """
        table_name = self._table_name(memory_type)
        try:
            table = self.db.open_table(table_name)

            # Filter out protected fields
            clean_updates = {
                k: v for k, v in updates.items()
                if k not in ("id", "vector")
            }

            if not clean_updates:
                return False

            # Add updated_at timestamp
            clean_updates["updated_at"] = datetime.now(timezone.utc).isoformat()

            table.update(where=f"id = '{record_id}'", values=clean_updates)
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    def delete(self, record_id: str, memory_type: str) -> bool:
        """Delete a memory record."""
        table_name = self._table_name(memory_type)
        try:
            table = self.db.open_table(table_name)
            table.delete(f"id = '{record_id}'")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def list_tables(self) -> list[str]:
        """List all memory tables."""
        return [
            t for t in self._list_table_names()
            if t.startswith(config.MEMORY_TABLE_PREFIX)
        ]

    def count(self, memory_type: str) -> int:
        """Count records in a memory table."""
        table_name = self._table_name(memory_type)
        try:
            table = self.db.open_table(table_name)
            return table.count_rows()
        except Exception:
            return 0

    def stats(self) -> dict:
        """Get stats for all memory tables."""
        tables = self.list_tables()
        return {
            t.replace(config.MEMORY_TABLE_PREFIX, ""): self.db.open_table(t).count_rows()
            for t in tables
        }
