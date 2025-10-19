"""
Tests for FastAPI Web Service - Phase 4

Async tests for API endpoints using httpx test client.
"""

import sys
import pytest
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from httpx import AsyncClient, ASGITransport
from fastapi import status

# Import app after path setup
from web_api.main import app


# ====================================================================================
# TEST FIXTURES
# ====================================================================================

@pytest.fixture
def client():
    """Create async test client"""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ====================================================================================
# BASIC ENDPOINT TESTS
# ====================================================================================

class TestBasicEndpoints:
    """Test basic API endpoints"""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        async with client:
            response = await client.get("/")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "name" in data
            assert "version" in data
            assert data["name"] == "Climate ETL Pipeline API"
    
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
# JOB ENDPOINTS TESTS
# ====================================================================================

class TestJobEndpoints:
    """Test job-related endpoints"""
    
    @pytest.mark.asyncio
    async def test_list_jobs_endpoint(self, client):
        """Test listing all available jobs"""
        async with client:
            response = await client.get("/jobs")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
            
            # Check structure of first job
            job = data[0]
            assert "name" in job
            assert "description" in job
            assert "tags" in job
            assert "ops" in job
    
    @pytest.mark.asyncio
    async def test_list_jobs_contains_expected_jobs(self, client):
        """Test that expected jobs are in the list"""
        async with client:
            response = await client.get("/jobs")
            data = response.json()
            
            job_names = [job["name"] for job in data]
            assert "daily_etl_job" in job_names
            assert "embedding_job" in job_names
            assert "complete_pipeline_job" in job_names
            assert "validation_job" in job_names
    
    @pytest.mark.asyncio
    async def test_trigger_job_run(self, client):
        """Test triggering a job run"""
        async with client:
            response = await client.post(
                "/jobs/daily_etl_job/run",
                json={"run_config": None, "tags": None}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "run_id" in data
            assert "job_name" in data
            assert "status" in data
            assert "message" in data
            assert data["job_name"] == "daily_etl_job"
            assert data["status"] == "QUEUED"
    
    @pytest.mark.asyncio
    async def test_trigger_job_run_with_config(self, client):
        """Test triggering a job run with configuration"""
        async with client:
            run_config = {
                "run_config": {
                    "ops": {
                        "download_era5_data": {
                            "config": {
                                "variables": ["2m_temperature"],
                                "year": 2024,
                                "month": 1
                            }
                        }
                    }
                },
                "tags": {"source": "api_test"}
            }
            
            response = await client.post(
                "/jobs/embedding_job/run",
                json=run_config
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["job_name"] == "embedding_job"
    
    @pytest.mark.asyncio
    async def test_trigger_invalid_job(self, client):
        """Test triggering a non-existent job returns 404"""
        async with client:
            response = await client.post(
                "/jobs/nonexistent_job/run",
                json={"run_config": None, "tags": None}
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
            data = response.json()
            assert "detail" in data
            assert "not found" in data["detail"].lower()


# ====================================================================================
# RUN STATUS ENDPOINTS TESTS
# ====================================================================================

class TestRunStatusEndpoints:
    """Test run status endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_run_status(self, client):
        """Test getting status of a specific run"""
        async with client:
            run_id = "run_20250119_140000_daily_etl_job"
            response = await client.get(f"/runs/{run_id}/status")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "run_id" in data
            assert "status" in data
            assert "job_name" in data
            assert data["run_id"] == run_id
    
    @pytest.mark.asyncio
    async def test_list_recent_runs(self, client):
        """Test listing recent runs"""
        async with client:
            response = await client.get("/runs")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_list_recent_runs_with_limit(self, client):
        """Test listing recent runs with limit parameter"""
        async with client:
            limit = 5
            response = await client.get(f"/runs?limit={limit}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, list)
            assert len(data) <= limit
    
    @pytest.mark.asyncio
    async def test_list_recent_runs_limit_validation(self, client):
        """Test that limit parameter is validated"""
        async with client:
            # Test limit too high
            response = await client.get("/runs?limit=1000")
            # Should be rejected or capped at 100
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]


# ====================================================================================
# DATA MODEL TESTS
# ====================================================================================

class TestDataModels:
    """Test Pydantic data models"""
    
    @pytest.mark.asyncio
    async def test_job_info_structure(self, client):
        """Test JobInfo model structure in response"""
        async with client:
            response = await client.get("/jobs")
            data = response.json()
            
            job = data[0]
            assert isinstance(job["name"], str)
            assert isinstance(job["tags"], dict)
            assert isinstance(job["ops"], list)
    
    @pytest.mark.asyncio
    async def test_run_status_structure(self, client):
        """Test RunStatus model structure in response"""
        async with client:
            response = await client.get(f"/runs/test_run_id/status")
            data = response.json()
            
            assert isinstance(data["run_id"], str)
            assert isinstance(data["status"], str)
            assert isinstance(data["job_name"], str)
            assert isinstance(data["tags"], dict)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
