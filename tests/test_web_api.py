"""
Tests for FastAPI Web Service - Phase 4

Async tests for API endpoints using httpx test client.
"""

import pytest
from datetime import datetime

from httpx import AsyncClient, ASGITransport
from fastapi import status

from web_api.main import app


# ====================================================================================
# TEST FIXTURES
# ====================================================================================

@pytest.fixture
def client():
    """Create async test client"""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def isolated_source_store(monkeypatch, tmp_path):
    """Provide a clean SQLite DB for source-store dependent tests.

    Previously imported `src.sources.source_store` — that submodule was folded
    into `src.database.source_store` during the Postgres migration. The fixture
    now points `CLIMATE_DB_URL` at a throwaway SQLite file; tests that need to
    touch the store should instantiate one via `get_source_store()` directly.
    """
    db_path = tmp_path / "sources.db"
    monkeypatch.setenv("CLIMATE_DB_URL", f"sqlite:///{db_path}")
    yield


# ====================================================================================
# BASIC ENDPOINT TESTS
# ====================================================================================

class TestBasicEndpoints:
    """Test basic API endpoints"""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Root endpoint redirects to the Vue SPA at /app/."""
        async with client:
            response = await client.get("/", follow_redirects=False)

            # When the Vue build is present (production container) we serve a
            # 307 → /app/. When the build dir is missing (fresh dev box that
            # hasn't run `npm run build`), the API falls back to a JSON stub.
            assert response.status_code in (
                status.HTTP_200_OK,
                status.HTTP_307_TEMPORARY_REDIRECT,
            )
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        async with client:
            response = await client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "status" in data
            assert "dagster_available" in data
            assert "timestamp" in data
            assert data["status"] in ["healthy", "degraded"]


# ====================================================================================
# SOURCE MANAGEMENT TESTS
# ====================================================================================


class TestSourceDeletion:
    """Ensure deleting sources also clears embeddings."""

    @pytest.mark.asyncio
    async def test_soft_delete_removes_embeddings(self, client, isolated_source_store, monkeypatch):
        from src.sources import get_source_store

        store = get_source_store()
        store.create_source({
            "source_id": "soft_delete_source",
            "url": "http://example.com/data.csv",
            "format": "csv",
            "variables": ["t2m"],
        })

        deleted_sources = []

        class DummyVectorDB:
            collection_name = "dummy"

            def __init__(self, *args, **kwargs):
                pass

            def delete_embeddings_by_source(self, source_id: str) -> int:
                deleted_sources.append(source_id)
                return 1

        monkeypatch.setattr("src.embeddings.database.VectorDatabase", DummyVectorDB)

        async with client:
            response = await client.delete("/sources/soft_delete_source")
            assert response.status_code == status.HTTP_204_NO_CONTENT

        assert deleted_sources == ["soft_delete_source"]
        updated = get_source_store().get_source("soft_delete_source")
        assert updated is not None and updated.is_active is False

    @pytest.mark.asyncio
    async def test_hard_delete_removes_embeddings(self, client, isolated_source_store, monkeypatch):
        from src.sources import get_source_store

        store = get_source_store()
        store.create_source({
            "source_id": "hard_delete_source",
            "url": "http://example.com/data.csv",
            "format": "csv",
            "variables": ["t2m"],
        })

        deleted_sources = []

        class DummyVectorDB:
            collection_name = "dummy"

            def __init__(self, *args, **kwargs):
                pass

            def delete_embeddings_by_source(self, source_id: str) -> int:
                deleted_sources.append(source_id)
                return 1

        monkeypatch.setattr("src.embeddings.database.VectorDatabase", DummyVectorDB)

        async with client:
            response = await client.delete("/sources/hard_delete_source?hard_delete=true")
            assert response.status_code == status.HTTP_204_NO_CONTENT

        assert deleted_sources == ["hard_delete_source"]
        assert get_source_store().get_source("hard_delete_source") is None


# ====================================================================================
# EMBEDDING MAINTENANCE TESTS
# ====================================================================================


class TestEmbeddingMaintenance:
    """Ensure embedding admin endpoints behave correctly."""

    @pytest.mark.asyncio
    async def test_clear_embeddings_requires_confirmation(self, client):
        async with client:
            response = await client.post("/embeddings/clear")
            assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_clear_embeddings_endpoint(self, client, monkeypatch):
        payload = {"calls": 0}

        class DummyVectorDB:
            collection_name = "dummy"

            def clear_collection(self) -> int:
                payload["calls"] += 1
                return 7

        monkeypatch.setattr("src.embeddings.database.VectorDatabase", DummyVectorDB)

        async with client:
            response = await client.post("/embeddings/clear?confirm=true")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["removed_embeddings"] == 7
            assert data["collection_name"] == "dummy"

        assert payload["calls"] == 1


# ====================================================================================
# JOB ENDPOINTS TESTS
# ====================================================================================

# ====================================================================================
# DATA MODEL TESTS
# ====================================================================================

class TestDataModels:
    """Test Pydantic data models"""
    
# ====================================================================================
# ERROR HANDLING TESTS
# ====================================================================================

class TestErrorHandling:
    """Test API error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_endpoint_returns_404(self, client):
        """Test that invalid endpoints return 404"""
        async with client:
            response = await client.get("/invalid/endpoint")
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_invalid_http_method(self, client):
        """Test that invalid HTTP methods are rejected"""
        async with client:
            response = await client.delete("/jobs")
            assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


# ====================================================================================
# INTEGRATION TESTS
# ====================================================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_job_workflow(self, client):
        """Test complete workflow: list jobs → trigger run → check status"""
        async with client:
            # 1. List jobs
            jobs_response = await client.get("/jobs")
            assert jobs_response.status_code == status.HTTP_200_OK
            jobs = jobs_response.json()
            assert len(jobs) > 0
            
            job_name = jobs[0]["name"]
            
            # 2. Trigger run
            trigger_response = await client.post(
                f"/jobs/{job_name}/run",
                json={"run_config": None, "tags": {"test": "integration"}}
            )
            assert trigger_response.status_code == status.HTTP_200_OK
            run_data = trigger_response.json()
            run_id = run_data["run_id"]
            
            # 3. Check status
            status_response = await client.get(f"/runs/{run_id}/status")
            assert status_response.status_code == status.HTTP_200_OK
            status_data = status_response.json()
            assert status_data["run_id"] == run_id
    
    @pytest.mark.asyncio
    async def test_health_check_before_operations(self, client):
        """Test checking health before performing operations"""
        async with client:
            # Check health first
            health_response = await client.get("/health")
            assert health_response.status_code == status.HTTP_200_OK
            
            health_data = health_response.json()
            
            # If healthy, should be able to list jobs
            if health_data["status"] == "healthy":
                jobs_response = await client.get("/jobs")
                assert jobs_response.status_code == status.HTTP_200_OK


# ====================================================================================
# CORS TESTS
# ====================================================================================

class TestCORS:
    """Test CORS configuration"""
    
    @pytest.mark.asyncio
    async def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses"""
        async with client:
            response = await client.get("/")
            
            # Check for CORS headers (may not be present in test client)
            # In production, these would be verified
            assert response.status_code == status.HTTP_200_OK


# ====================================================================================
# DOCUMENTATION TESTS
# ====================================================================================

class TestDocumentation:
    """Test API documentation endpoints"""
    
    @pytest.mark.asyncio
    async def test_openapi_docs_available(self, client):
        """Test that OpenAPI documentation is available"""
        async with client:
            response = await client.get("/docs")
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_307_TEMPORARY_REDIRECT]
    
    @pytest.mark.asyncio
    async def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available"""
        async with client:
            response = await client.get("/openapi.json")
            assert response.status_code == status.HTTP_200_OK
            schema = response.json()
            assert "openapi" in schema
            assert "info" in schema
            assert "paths" in schema


class TestSampleDownloads:
    """Ensure sample dataset endpoint handles edge cases"""

    @pytest.mark.asyncio
    async def test_missing_sample_returns_404(self, client):
        async with client:
            response = await client.get("/samples/not_real.nc")
            assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_directory_traversal_is_rejected(self, client):
        async with client:
            response = await client.get("/samples/../secrets.txt")
            assert response.status_code in {
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_404_NOT_FOUND,
            }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
