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
Configuration for SQLite database pool.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PoolConfig:
    """
    Configuration for database connection pool.

    Attributes:
        db_dir: Root directory containing database folders (e.g., "sqlite_databases_test")
        db_ids: List of database IDs to pool (e.g., ["address", "airline", ...])
        pool_size_per_db: Number of copies to create per database (default: 2)
        template_suffix: Suffix for template databases (default: "_template.sqlite")
        working_prefix: Prefix for working copies (default: "pool")
        max_wait_seconds: Max seconds to wait for available DB (default: 30)
        cleanup_on_release: Whether to delete working copy on release (default: False)
        reset_on_acquire_ephemeral: Whether to reset DB when acquiring in ephemeral mode (default: True)

    Example:
        config = PoolConfig(
            db_dir="/path/to/sqlite_databases_test",
            db_ids=["address", "airline", "book_publishing_company"],
            pool_size_per_db=3,
            max_wait_seconds=60
        )
    """
    db_dir: str
    db_ids: List[str]
    pool_size_per_db: int = 2
    pool_start_index: int = 0  # Start index for pool file naming (for worker partitioning)
    template_suffix: str = "_template.sqlite"
    working_prefix: str = "pool"
    max_wait_seconds: float = 30.0
    cleanup_on_release: bool = False
    reset_on_acquire_ephemeral: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not self.db_dir:
            raise ValueError("db_dir cannot be empty")

        if not self.db_ids:
            raise ValueError("db_ids cannot be empty")

        if self.pool_size_per_db < 1:
            raise ValueError(f"pool_size_per_db must be >= 1, got {self.pool_size_per_db}")

        if self.max_wait_seconds <= 0:
            raise ValueError(f"max_wait_seconds must be > 0, got {self.max_wait_seconds}")

    def get_template_path(self, db_id: str) -> str:
        """Get template database path for given db_id."""
        import os
        return os.path.join(self.db_dir, db_id, f"{db_id}{self.template_suffix}")

    def get_working_dir(self, db_id: str) -> str:
        """Get working directory for given db_id."""
        import os
        return os.path.join(self.db_dir, db_id)
