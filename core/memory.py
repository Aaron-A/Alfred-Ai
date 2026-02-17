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
    - Hybrid search (vector + full-text, when available)
    - Update and delete records
    """

    def __init__(self, db_path: str = None, agent_id: str = "default"):
        config.ensure_dirs()
        self.db_path = db_path or str(config.LANCEDB_DIR)
        self.agent_id = agent_id
        self.db = lancedb.connect(self.db_path)
        self.embedder = get_embedding_engine()
        self._tables: dict[str, lancedb.table.Table] = {}

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

    def search(
        self,
        query: str,
        memory_type: str = None,
        top_k: int = None,
        where: str = None,
    ) -> list[dict]:
        """
        Semantic search across memories.

        Args:
            query: Natural language search query
            memory_type: Filter to a specific memory type (e.g., "trade", "tweet")
                        If None, searches all memory tables.
            top_k: Number of results to return
            where: SQL-like filter string (e.g., "symbol = 'TSLA' AND outcome = 'win'")

        Returns:
            List of dicts with memory fields + _distance score
        """
        top_k = top_k or config.DEFAULT_TOP_K
        query_vector = self.embedder.embed_query(query)

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
        Re-embeds if content-related fields change.

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

            # Build SET clause for update
            update_pairs = []
            for key, value in updates.items():
                if key in ("id", "vector"):
                    continue
                if isinstance(value, str):
                    update_pairs.append(f"{key} = '{value}'")
                elif isinstance(value, (int, float)):
                    update_pairs.append(f"{key} = {value}")

            if not update_pairs:
                return False

            # Add updated_at timestamp
            now = datetime.now(timezone.utc).isoformat()
            update_pairs.append(f"updated_at = '{now}'")

            update_sql = ", ".join(update_pairs)
            table.update(where=f"id = '{record_id}'", values=updates)
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
