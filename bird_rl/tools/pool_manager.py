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
Global SQLite connection pool for SQL debugging training.

This module initializes a shared database connection pool that is used by both
sql_executor_tool and submit_solution_tool. The pool is created once when this
module is first imported, and all tools share the same pool instance.

Design:
- Pool created at module import time (runs once)
- 13 unique databases from training data
- 16 copies per database = 208 total DB connections
- Thread-safe for concurrent access

Usage Modes:
- Persistent mode (tools during trajectory): State maintained, NO reset
- Ephemeral mode (reward function): Fresh DB state with reset
"""

import logging
import os
import sys

from bird_rl.database.pool import DatabasePool
from bird_rl.database.config import PoolConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Pool partitioning strategy:
# - Tools (AgentLoopWorker): Use indices [0-127] (4 actors × 32 copies each)
# - Reward (TaskRunner): Use indices [128-159] (32 dedicated copies)
# Total: 160 copies per database
#
# IMPORTANT: Each actor MUST have exclusive indices because:
# - Each actor has its OWN pool instance in memory
# - Actor A marking DB as in_use doesn't notify Actor B
# - Without partitioning, multiple actors could use same DB file

# Check if this is reward function (set USE_REWARD_POOL=1 in reward function)
USE_REWARD_POOL = os.getenv("USE_REWARD_POOL", "0") == "1"

if USE_REWARD_POOL:
    # Reward function pool: 32 copies at indices [128-159]
    pool_start_index = 128
    POOL_COPIES_PER_WORKER = 32
    logger.warning(f"REWARD POOL MODE: PID {os.getpid()} → using pool copies [128-159]")
else:
    # Tool pool: 128 copies at indices [0-127], partitioned among 4 actors
    NUM_ACTORS = 4
    POOL_COPIES_PER_ACTOR = 32  # 128 total / 4 actors = 32 per actor

    # Get actor index from Ray actor name
    # AgentLoopWorker is created with name=f"agent_loop_worker_{i}" where i=0,1,2,3
    actor_index = None

    try:
        import ray
        import re
        actor_name = ray.get_runtime_context().get_actor_name()
        if actor_name:
            # Extract index from name like "agent_loop_worker_0_abc12345"
            match = re.match(r'agent_loop_worker_(\d+)', actor_name)
            if match:
                actor_index = int(match.group(1)) % NUM_ACTORS
                logger.warning(f"TOOL POOL MODE: PID {os.getpid()} → actor={actor_name} → actor_index={actor_index}")
    except Exception as e:
        logger.warning(f"TOOL POOL MODE: PID {os.getpid()} → failed to get actor name: {e}")

    if actor_index is None:
        # Fallback: PID-based (may cause collisions, but better than nothing)
        actor_index = os.getpid() % NUM_ACTORS
        logger.warning(f"TOOL POOL MODE: PID {os.getpid()} → fallback to PID-based index {actor_index} (WARNING: may collide)")

    # Calculate which pool indices this actor should use
    pool_start_index = actor_index * POOL_COPIES_PER_ACTOR
    POOL_COPIES_PER_WORKER = POOL_COPIES_PER_ACTOR  # Keep variable name for compatibility

pool_end_index = pool_start_index + POOL_COPIES_PER_WORKER
logger.warning(f"Worker PID {os.getpid()} → using pool copies [{pool_start_index}, {pool_end_index})")

# Training database IDs (all databases in test set)
TRAIN_DB_IDS = [
    'address',
    'airline',
    'book_publishing_company',
    'books',
    'california_schools',
    'car_retails',
    'card_games',
    'chinook',
    'codebase_community',
    'debit_card_specializing',
    'employees',
    'erolp',
    'esophageal',
    'european_football_2',
    'financial',
    'formula_1',
    'global_atlas',
    'hockey',
    'lego',
    'movie_3',
    'netflix',
    'olympics',
    'public_review_platform',
    'spotify',
    'student_club',
    'superhero',
    'thrombosis_prediction',
    'toxicology'
]

# Database directory (from SQL_REWARD_DB_DIR environment variable)
DB_DIR = os.getenv("SQL_REWARD_DB_DIR", "")

# Pool configuration - each worker gets a subset of pool copies
config = PoolConfig(
    db_dir=DB_DIR,
    db_ids=TRAIN_DB_IDS,
    pool_size_per_db=POOL_COPIES_PER_WORKER,  # Only create copies for this worker
    pool_start_index=pool_start_index,  # Start index for pool naming
    max_wait_seconds=10.0,  # Fail fast: DBs are held per-trajectory, waiting rarely helps
    reset_on_acquire_ephemeral=True,  # Legacy flag (now always resets on acquire)
    cleanup_on_release=False  # No reset on release - reset happens on acquire instead
)

# Initialize pool (runs ONCE when this module is first imported)
pool_mode = "REWARD" if USE_REWARD_POOL else "TOOL"
logger.info(f"Initializing SQLite connection pool ({pool_mode} mode)...")
logger.info(f"  DB directory: {DB_DIR}")
logger.info(f"  Databases: {len(TRAIN_DB_IDS)}")
logger.info(f"  Pool copies: {config.pool_size_per_db} per DB (indices {pool_start_index}-{pool_end_index-1})")
logger.info(f"  Total DB copies: {len(TRAIN_DB_IDS) * config.pool_size_per_db}")

pool = DatabasePool(config)

# Shared state for tracking which instance_id uses which pooled_db
# This ensures both sql_executor_tool and submit_solution_tool share the same DB
# Format: {instance_id: {"pooled_db": PooledDatabase, "ref_count": int}}
instance_db_map = {}

logger.info(f"✓ Pool initialized successfully ({pool_mode} mode)")
logger.info(f"  Pool stats: {pool.get_stats()}")
