import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def project_root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def db_path(project_root: str) -> str:
    return os.path.join(project_root, "data", "northwind.sqlite")


@pytest.fixture(autouse=True)
def set_env_db(db_path: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NORTHWIND_DB", db_path)
    yield

