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
Pooled database connection wrapper.
"""
import os
import shutil
import sqlite3
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class PooledDatabase:
    """
    Wrapper for a pooled database copy.

    This represents a single working copy of a template database in the pool.
    It manages:
    - Physical database file path
    - SQLite connection
    - Reset operations from template
    - Cleanup

    Attributes:
        db_id: Database identifier
        template_path: Path to read-only template database
        working_path: Path to this working copy
        connection: Active SQLite connection (None if closed)
        pool_index: Index in the pool (for identification)
        in_use: Whether this DB is currently acquired
        needs_reset: Whether this DB needs reset before next use (False after init/reset, True after release)
    """

    def __init__(
        self,
        db_id: str,
        template_path: str,
        working_path: str,
        pool_index: int
    ):
        """
        Initialize a pooled database.

        Args:
            db_id: Database identifier (e.g., "address")
            template_path: Path to template database
            working_path: Path to working copy (should not exist yet)
            pool_index: Index in pool (0-based)

        Raises:
            FileNotFoundError: If template_path doesn't exist
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template database not found: {template_path}")

        self.db_id = db_id
        self.template_path = template_path
        self.working_path = working_path
        self.pool_index = pool_index
        self.in_use = False
        self.needs_reset = True  # Always reset on first acquire to ensure clean state
        # (pool files from previous training runs may have stale preprocess tables)
        self.connection: Optional[sqlite3.Connection] = None

        # Create initial working copy
        # Use force_fresh=False since bash script pre-creates pool files
        # This avoids race conditions when multiple Ray workers initialize simultaneously
        self._copy_from_template(force_fresh=False)
        self._open_connection()

        logger.info(f"Created pooled database: {db_id}[{pool_index}] at {working_path}")

    def _copy_from_template(self, force_fresh: bool = False) -> None:
        """
        Copy template to working path using SQLite backup API.

        Args:
            force_fresh: If True, always create fresh copy (delete existing file first).
                        If False, reuse existing valid file (for Ray worker recovery).
        """
        # If force_fresh, delete existing file first (including WAL files!)
        if force_fresh:
            # Delete WAL files FIRST before main database file
            # This ensures no WAL contamination
            try:
                for suffix in ['-wal', '-shm', '-journal']:
                    wal_path = self.working_path + suffix
                    if os.path.exists(wal_path):
                        try:
                            os.remove(wal_path)
                        except OSError as e:
                            logger.warning(f"Could not delete {wal_path}: {e}")

                # Now delete main database file
                if os.path.exists(self.working_path):
                    os.remove(self.working_path)
            except (FileNotFoundError, OSError) as e:
                # If deletion fails, log but continue - the copy might still work
                logger.warning(f"Error during file deletion: {e}")

        # If working copy already exists and not force_fresh, reuse it
        if not force_fresh and os.path.exists(self.working_path):
            try:
                # Ensure write permissions (file might have wrong permissions)
                os.chmod(self.working_path, 0o644)

                # Test that the file is valid
                test_conn = sqlite3.connect(self.working_path, timeout=5.0, check_same_thread=False)
                test_conn.execute("SELECT 1")
                test_conn.close()
                logger.debug(f"Reusing existing working database at {self.working_path}")
                return
            except Exception as e:
                logger.warning(f"Existing working database corrupted, recreating: {e}")
                try:
                    os.remove(self.working_path)
                    for suffix in ['-wal', '-shm', '-journal']:
                        wal_path = self.working_path + suffix
                        if os.path.exists(wal_path):
                            os.remove(wal_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up corrupted database: {cleanup_error}")

        # Copy from template - SIMPLE file copy, no connection opening
        try:
            shutil.copyfile(self.template_path, self.working_path)
            # Ensure write permissions (template might be read-only)
            os.chmod(self.working_path, 0o644)  # rw-r--r--
            logger.debug(f"Copied template to {self.working_path}")
        except (OSError, FileNotFoundError) as e:
            # Check if another thread created it
            if os.path.exists(self.working_path):
                logger.debug(f"File created by another thread: {self.working_path}")
                return
            logger.error(f"Failed to copy template: {e}")
            raise

    def _open_connection(self) -> None:
        """Open SQLite connection in read-write mode."""
        if self.connection is not None:
            logger.warning("Connection already open, closing first")
            self._close_connection()

        # Explicitly open in read-write mode using URI
        # This prevents SQLite from falling back to read-only mode if it detects issues
        self.connection = sqlite3.connect(
            f"file:{self.working_path}?mode=rw",
            uri=True,
            timeout=30.0,
            check_same_thread=False
        )
        logger.debug(f"Opened connection to {self.working_path} in read-write mode")

    def _close_connection(self) -> None:
        """Close SQLite connection."""
        if self.connection is not None:
            try:
                self.connection.commit()
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self.connection = None

    def reset(self) -> None:
        """
        Reset database to template state using atomic file replacement.

        Uses copy-to-temp-then-atomic-rename to avoid race conditions.
        Includes retry logic with exponential backoff to handle transient
        disk I/O errors.

        CRITICAL: WAL files are deleted BEFORE replacement to prevent
        mismatch between new database and old WAL files.
        """
        logger.debug(f"Resetting {self.db_id}[{self.pool_index}]")

        max_retries = 3
        temp_path = None

        for attempt in range(max_retries):
            try:
                # Step 1: Copy template to a TEMPORARY file (doesn't touch working_path)
                # Use unique name to avoid conflicts between threads
                temp_path = f"{self.working_path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
                shutil.copyfile(self.template_path, temp_path)

                # CRITICAL: Ensure write permissions on the copy
                # Template might be read-only, but working copy needs write access
                os.chmod(temp_path, 0o644)  # rw-r--r--

                # Step 2: Close connection to release locks
                self._close_connection()

                # Step 3: Delete WAL/SHM/journal files BEFORE replacement (CRITICAL!)
                # This prevents mismatch where new DB file has old WAL files applied
                for suffix in ['-wal', '-shm', '-journal']:
                    wal_path = self.working_path + suffix
                    try:
                        if os.path.exists(wal_path):
                            os.remove(wal_path)
                    except OSError:
                        pass  # Already deleted or doesn't exist

                # Step 4: Atomically replace working file with temp file
                # os.replace() is atomic on Linux - no window where file is missing
                os.replace(temp_path, self.working_path)
                temp_path = None  # Successfully moved, don't clean up

                # Step 5: Reopen connection with fresh database
                self._open_connection()

                # Mark as clean - no reset needed until next use
                self.needs_reset = False

                logger.debug(f"Reset complete for {self.db_id}[{self.pool_index}]")
                return  # Success!

            except (OSError, IOError) as e:
                # Transient I/O error - retry with backoff
                logger.warning(
                    f"Reset attempt {attempt + 1}/{max_retries} failed for "
                    f"{self.db_id}[{self.pool_index}]: {e}"
                )

                # Cleanup temp file if it still exists
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass

                # If this was the last retry, give up
                if attempt == max_retries - 1:
                    logger.error(f"Reset failed after {max_retries} attempts")
                    # Try to recover by ensuring working file exists
                    try:
                        if not os.path.exists(self.working_path):
                            shutil.copyfile(self.template_path, self.working_path)
                        self._open_connection()
                    except:
                        pass
                    raise

                # Exponential backoff: 10ms, 20ms, 40ms
                time.sleep(0.01 * (2 ** attempt))

            except Exception as e:
                # Non-I/O error - don't retry
                logger.error(f"Reset failed for {self.db_id}[{self.pool_index}]: {e}")
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise

    def execute_preprocess(self, preprocess_sql_list: list[str]) -> None:
        """
        Execute preprocessing SQL statements.

        Args:
            preprocess_sql_list: List of SQL statements to execute

        Raises:
            RuntimeError: If connection is not open
            sqlite3.Error: If SQL execution fails
        """
        if self.connection is None:
            raise RuntimeError("Connection is not open")

        if not preprocess_sql_list:
            return

        logger.debug(f"Executing {len(preprocess_sql_list)} preprocess SQLs")

        cursor = None
        try:
            cursor = self.connection.cursor()
            for sql in preprocess_sql_list:
                cursor.execute(sql)
            self.connection.commit()
            cursor.close()
            cursor = None

            logger.debug("Preprocess complete")

        except Exception as e:
            # Close cursor to prevent resource leak
            if cursor is not None:
                try:
                    cursor.close()
                except:
                    pass
            # logger.error(f"Preprocess SQL failed for {self.db_id}[{self.pool_index}]: {e}")
            raise

    def cleanup(self) -> None:
        """
        Cleanup resources.

        This:
        1. Closes connection
        2. Deletes working copy and WAL files

        Call this when shutting down the pool.
        """
        logger.debug(f"Cleaning up {self.db_id}[{self.pool_index}]")

        self._close_connection()

        # Delete working copy
        if os.path.exists(self.working_path):
            try:
                os.remove(self.working_path)
                # Clean up WAL files
                for suffix in ['-wal', '-shm', '-journal']:
                    wal_path = self.working_path + suffix
                    if os.path.exists(wal_path):
                        os.remove(wal_path)
                logger.debug(f"Deleted working copy: {self.working_path}")
            except Exception as e:
                logger.warning(f"Error deleting working copy: {e}")

    def __repr__(self) -> str:
        status = "in_use" if self.in_use else "available"
        conn_status = "connected" if self.connection else "closed"
        return f"<PooledDatabase {self.db_id}[{self.pool_index}] {status} {conn_status}>"
