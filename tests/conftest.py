from __future__ import annotations

from pathlib import Path

import pytest

from travel_agent.config import Settings
from travel_agent.persistence import Database


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.sqlite3")
    yield database
    database.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(database_path=tmp_path / "test.sqlite3", output_dir=tmp_path / "output")
