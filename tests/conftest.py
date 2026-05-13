import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Provides a TestClient for the main FastAPI application."""
    return TestClient(app)
