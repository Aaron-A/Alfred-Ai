"""
Alfred AI - Memory Layer
LanceDB-backed vector memory with typed schemas and hybrid search.

This is the brain of Alfred. Every agent reads from and writes to this
shared memory layer. Memories are embedded locally, stored in LanceDB,
and retrievable via semantic search + metadata filtering.
"""

import time
from datetime import datetime, timezone
from typing import Optional, Type
from dataclasses import asdict

import lancedb
import pyarrow as pa

from .config import config
from .embeddings import get_embedding_engine
from models.base import MemoryRecord


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

    def _table_name(self, memory_type: str) -> str:
        """Generate table name from memory type."""
        return f"{config.MEMORY_TABLE_PREFIX}{memory_type}"

    def _get_or_create_table(self, data: list[dict]) -> lancedb.table.Table:
        """Get existing table or create one from data."""
        # Infer memory_type from the first record
        memory_type = data[0].get("memory_type", "generic")
        table_name = self._table_name(memory_type)

        if table_name in self._tables:
            return self._tables[table_name], False

        existing_tables = self.db.table_names()

        if table_name in existing_tables:
            table = self.db.open_table(table_name)
            self._tables[table_name] = table
            return table, False
        else:
            # Create table with the full batch of data
            table = self.db.create_table(table_name, data=data)
            print(f"[alfred] Created memory table: {table_name}")
            self._tables[table_name] = table
            return table, True  # True = data was already inserted via create

    def store(self, record: MemoryRecord) -> str:
        """
        Store a memory record with automatic embedding.

        Args:
            record: Any MemoryRecord subclass (TradeMemory, TweetMemory, etc.)

        Returns:
            The record ID
        """
        # Set agent_id if not already set
        if not record.agent_id:
            record.agent_id = self.agent_id

        # Generate embedding from the rich text representation
        embed_text = record.to_embed_text()
        vector = self.embedder.embed_documents([embed_text])[0]

        # Convert to dict and add vector
        data = record.to_dict()
        data["vector"] = vector

        # Get or create table
        table, already_inserted = self._get_or_create_table([data])

        # Only add if table already existed (data not yet inserted)
        if not already_inserted:
            table.add([data])

        return record.id

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

        Returns:
            List of dicts with memory fields + _relevance_score (hybrid) or _distance (vector)
        """
        top_k = top_k or config.DEFAULT_TOP_K
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
                t for t in self.db.table_names()
                if t.startswith(config.MEMORY_TABLE_PREFIX)
            ]

        for table_name in table_names:
            try:
                table = self.db.open_table(table_name)

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
                print(f"[alfred] Warning: search failed on {table_name}: {e}")

        # Sort by distance (lower = more similar) and trim to top_k
        results.sort(key=lambda x: x.get("_distance", float("inf")))
        return results[:top_k]

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
            print(f"[alfred] Update failed: {e}")
            return False

    def delete(self, record_id: str, memory_type: str) -> bool:
        """Delete a memory record."""
        table_name = self._table_name(memory_type)
        try:
            table = self.db.open_table(table_name)
            table.delete(f"id = '{record_id}'")
            return True
        except Exception as e:
            print(f"[alfred] Delete failed: {e}")
            return False

    def list_tables(self) -> list[str]:
        """List all memory tables."""
        return [
            t for t in self.db.table_names()
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
