# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SQLite database connection pool.

This module provides a thread-safe connection pool for SQLite databases.
It pre-creates working copies of template databases and manages their lifecycle.
"""
import os
import threading
import time
import logging
from typing import Optional, List, Literal
from contextlib import contextmanager

from .config import PoolConfig
from .connection import PooledDatabase

logger = logging.getLogger(__name__)


class DatabasePool:
    """
    Thread-safe connection pool for SQLite databases.

    This pool:
    1. Pre-creates fixed number of working copies per database
    2. Manages acquisition/release with thread-safe locking
    3. Supports two modes:
       - 'persistent': Reuse existing DB state (for rollout phase)
       - 'ephemeral': Reset DB to clean state (for test phase)
    4. Provides context manager interface for automatic release

    Example:
        # Initialize pool
        config = PoolConfig(
            db_dir="sqlite_databases_test",
            db_ids=["address", "airline"],
            pool_size_per_db=2
        )
        pool = DatabasePool(config)

        # Use persistent mode (rollout phase)
        with pool.acquire("address", mode="persistent") as db:
            db.connection.execute("CREATE TEMP TABLE foo (x INT)")
            # ... multiple SQL executions maintaining state ...

        # Use ephemeral mode (test phase)
        with pool.acquire("address", mode="ephemeral") as db:
            # DB is reset to template state
            result = db.connection.execute("SELECT * FROM users").fetchall()

        # Cleanup
        pool.close()

    Thread Safety:
        All operations are thread-safe. Multiple threads can acquire/release
        databases concurrently.

    Resource Management:
        - Disk usage: pool_size_per_db × num_databases × avg_db_size
        - Memory: One SQLite connection per pooled database
        - File handles: One per active connection
    """

    def __init__(self, config: PoolConfig):
        """
        Initialize the database pool.

        This creates all working copies immediately (pre-allocation).

        Args:
            config: Pool configuration

        Raises:
            FileNotFoundError: If template database doesn't exist
            ValueError: If configuration is invalid
        """
        self.config = config
        self._pools: dict[str, List[PooledDatabase]] = {}
        self._locks: dict[str, threading.Condition] = {}
        self._closed = False
        self._global_lock = threading.Lock()

        # Initialize pools for each database
        for db_id in config.db_ids:
            self._init_db_pool(db_id)

        logger.info(
            f"DatabasePool initialized: {len(config.db_ids)} databases, "
            f"{config.pool_size_per_db} copies each = "
            f"{len(config.db_ids) * config.pool_size_per_db} total working copies"
        )

    def _init_db_pool(self, db_id: str) -> None:
        """
        Initialize pool for a single database.

        Args:
            db_id: Database identifier
        """
        template_path = self.config.get_template_path(db_id)
        working_dir = self.config.get_working_dir(db_id)

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template database not found: {template_path}")

        # Create pool list and lock for this database
        self._pools[db_id] = []
        self._locks[db_id] = threading.Condition(threading.Lock())

        # Pre-create working copies (use pool_start_index for worker partitioning)
        for i in range(self.config.pool_size_per_db):
            # Calculate actual pool index (accounts for worker partitioning)
            actual_index = self.config.pool_start_index + i

            working_path = os.path.join(
                working_dir,
                f"{db_id}_{self.config.working_prefix}_{actual_index}.sqlite"
            )

            pooled_db = PooledDatabase(
                db_id=db_id,
                template_path=template_path,
                working_path=working_path,
                pool_index=actual_index
            )

            self._pools[db_id].append(pooled_db)

        logger.info(f"Initialized pool for {db_id}: {self.config.pool_size_per_db} copies (indices {self.config.pool_start_index}-{self.config.pool_start_index + self.config.pool_size_per_db - 1})")

    def acquire(
        self,
        db_id: str,
        mode: Literal["persistent", "ephemeral"] = "persistent",
        timeout: Optional[float] = None,
        preprocess_sql: Optional[List[str]] = None
    ) -> PooledDatabase:
        """
        Acquire a database from the pool.

        This blocks until a database is available or timeout expires.

        Args:
            db_id: Database identifier
            mode: Acquisition mode:
                - "persistent": Reuse existing state (default)
                - "ephemeral": Reset to template before returning
            timeout: Max seconds to wait (default: config.max_wait_seconds)
            preprocess_sql: Optional SQL statements to execute after acquisition/reset

        Returns:
            PooledDatabase ready for use

        Raises:
            ValueError: If db_id not in pool
            TimeoutError: If timeout expires before DB available
            RuntimeError: If pool is closed

        Example:
            # Manual management
            db = pool.acquire("address", mode="ephemeral")
            try:
                result = db.connection.execute("SELECT * FROM users")
            finally:
                pool.release(db)

            # Context manager (recommended)
            with pool.acquire("address", mode="ephemeral") as db:
                result = db.connection.execute("SELECT * FROM users")
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        if db_id not in self._pools:
            raise ValueError(f"Database {db_id} not in pool. Available: {list(self._pools.keys())}")

        if timeout is None:
            timeout = self.config.max_wait_seconds

        start_time = time.time()
        condition = self._locks[db_id]

        # Find and claim an available database
        pooled_db = None
        with condition:
            while True:
                # Try to find available database
                available_count = sum(1 for db in self._pools[db_id] if not db.in_use)
                in_use_count = sum(1 for db in self._pools[db_id] if db.in_use)
                logger.debug(f"[POOL ACQUIRE] {db_id}: {available_count} available, {in_use_count} in use")

                for db in self._pools[db_id]:
                    if not db.in_use:
                        # Found available DB - mark as in use
                        db.in_use = True
                        pooled_db = db
                        # logger.warning(f"[POOL ACQUIRE] ✓ Claimed {db_id}[{db.pool_index}], marking in_use=True")
                        break

                if pooled_db:
                    break  # Exit the while loop, we found a DB

                # No available DB, check timeout
                elapsed = time.time() - start_time
                remaining = timeout - elapsed

                if remaining <= 0:
                    # Log which pool indices are in use for debugging
                    in_use_indices = [db.pool_index for db in self._pools[db_id] if db.in_use]
                    logger.error(
                        f"[POOL TIMEOUT] {db_id} timeout after {timeout:.1f}s. "
                        f"Pool size: {self.config.pool_size_per_db}, "
                        f"in_use indices: {in_use_indices}"
                    )
                    raise TimeoutError(
                        f"Timeout waiting for {db_id} after {timeout:.1f}s. "
                        f"Pool size: {self.config.pool_size_per_db}, "
                        f"all copies in use."
                    )

                # Wait for release notification
                logger.debug(f"Waiting for {db_id} (timeout={remaining:.1f}s)")
                condition.wait(timeout=remaining)

        # Reset OUTSIDE the lock (DB stays in_use=True so no one else takes it)
        # Only reset if the database has been used (needs_reset=True)
        # Fresh databases from initialization don't need reset
        if pooled_db.needs_reset:
            try:
                pooled_db.reset()
                logger.debug(f"Reset complete for {db_id}[{pooled_db.pool_index}]")
            except Exception as e:
                # Reset failed - release the DB and raise
                with condition:
                    pooled_db.in_use = False
                    condition.notify()
                raise
        else:
            logger.debug(f"Skipping reset for {db_id}[{pooled_db.pool_index}] (fresh database)")

        # Execute preprocessing SQL if provided (after reset)
        # Errors are logged but NOT propagated — the DB is still usable.
        # Common case: "table already exists" from stale pool copies.
        if preprocess_sql:
            try:
                pooled_db.execute_preprocess(preprocess_sql)
            except Exception as e:
                # logger.warning(
                #     f"[POOL ACQUIRE] Preprocess SQL failed for {db_id}[{pooled_db.pool_index}]: {e}. "
                #     f"Continuing with DB as-is (trajectory can still proceed)."
                # )
                pass

        logger.debug(f"Acquired {db_id}[{pooled_db.pool_index}] in {mode} mode")
        return pooled_db

    def release(self, pooled_db: PooledDatabase) -> None:
        """
        Release a database back to the pool.

        Simply marks the database as available for reuse. Reset happens
        on the NEXT acquire (not during release) to avoid errors during
        release and ensure clean state at start of trajectory.

        Args:
            pooled_db: Database to release

        Raises:
            ValueError: If database doesn't belong to this pool
            RuntimeError: If database is not in use
        """
        if self._closed:
            logger.warning("Releasing database to closed pool")
            return

        db_id = pooled_db.db_id

        if db_id not in self._pools:
            raise ValueError(f"Database {db_id} doesn't belong to this pool")

        if pooled_db not in self._pools[db_id]:
            raise ValueError(f"This specific PooledDatabase instance doesn't belong to this pool")

        condition = self._locks[db_id]

        with condition:
            if not pooled_db.in_use:
                logger.error(f"[POOL RELEASE] ❌ {db_id}[{pooled_db.pool_index}] is NOT in use! Cannot release.")
                raise RuntimeError(f"{pooled_db} is not in use, cannot release")

            # Mark as needing reset for next acquire (database has been used)
            pooled_db.needs_reset = True

            # Mark as available - reset will happen on next acquire
            pooled_db.in_use = False

            # Count pool state after release
            available_count = sum(1 for db in self._pools[db_id] if not db.in_use)
            in_use_count = sum(1 for db in self._pools[db_id] if db.in_use)

            # logger.warning(f"[POOL RELEASE] ✓ Released {db_id}[{pooled_db.pool_index}], marked in_use=False")
            # logger.warning(f"[POOL RELEASE] Pool state: {available_count} available, {in_use_count} in use")
            # print(f"[POOL RELEASE] ✓ {db_id}[{pooled_db.pool_index}] in_use=False, pool now: {available_count} avail/{in_use_count} in_use", flush=True)

            # Notify waiting threads
            condition.notify()

    @contextmanager
    def acquire_context(
        self,
        db_id: str,
        mode: Literal["persistent", "ephemeral"] = "persistent",
        timeout: Optional[float] = None,
        preprocess_sql: Optional[List[str]] = None
    ):
        """
        Context manager for acquiring and auto-releasing a database.

        Args:
            db_id: Database identifier
            mode: Acquisition mode ("persistent" or "ephemeral")
            timeout: Max seconds to wait
            preprocess_sql: Optional SQL statements to execute after acquisition/reset

        Yields:
            PooledDatabase ready for use

        Example:
            with pool.acquire_context("address", mode="ephemeral") as db:
                result = db.connection.execute("SELECT * FROM users")
                # DB automatically released when exiting with block
        """
        pooled_db = self.acquire(db_id, mode=mode, timeout=timeout, preprocess_sql=preprocess_sql)
        try:
            yield pooled_db
        finally:
            self.release(pooled_db)

    def get_stats(self, db_id: Optional[str] = None) -> dict:
        """
        Get pool statistics.

        Args:
            db_id: If specified, get stats for specific database.
                   Otherwise, get stats for all databases.

        Returns:
            Dictionary with pool statistics

        Example:
            stats = pool.get_stats("address")
            print(f"In use: {stats['in_use']}/{stats['total']}")
        """
        if db_id is not None:
            if db_id not in self._pools:
                raise ValueError(f"Database {db_id} not in pool")

            pool = self._pools[db_id]
            in_use = sum(1 for db in pool if db.in_use)
            return {
                "db_id": db_id,
                "total": len(pool),
                "in_use": in_use,
                "available": len(pool) - in_use,
                "utilization": in_use / len(pool) if pool else 0.0
            }
        else:
            # Aggregate stats
            total = sum(len(pool) for pool in self._pools.values())
            in_use = sum(1 for pool in self._pools.values() for db in pool if db.in_use)
            return {
                "total_databases": len(self._pools),
                "total_copies": total,
                "in_use": in_use,
                "available": total - in_use,
                "utilization": in_use / total if total else 0.0,
                "per_database": {
                    db_id: self.get_stats(db_id)
                    for db_id in self._pools.keys()
                }
            }

    def close(self) -> None:
        """
        Close the pool and cleanup all resources.

        This:
        1. Closes all database connections
        2. Deletes all working copies
        3. Marks pool as closed (further operations will raise RuntimeError)

        Note:
            This is not thread-safe with respect to acquire/release.
            Ensure all databases are released before calling close.
        """
        if self._closed:
            logger.warning("Pool already closed")
            return

        with self._global_lock:
            self._closed = True

            logger.info("Closing pool...")

            for db_id, pool in self._pools.items():
                for pooled_db in pool:
                    if pooled_db.in_use:
                        logger.warning(f"Closing {pooled_db} while still in use!")
                    pooled_db.cleanup()

            self._pools.clear()
            self._locks.clear()

            logger.info("Pool closed")

    def __enter__(self):
        """Support context manager for pool lifecycle."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-close pool on context exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        stats = self.get_stats()
        status = "closed" if self._closed else "open"
        return (
            f"<DatabasePool {status} "
            f"databases={stats['total_databases']} "
            f"copies={stats['total_copies']} "
            f"in_use={stats['in_use']}>"
        )
