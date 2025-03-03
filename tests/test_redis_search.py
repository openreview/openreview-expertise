import pytest
import json
import pickle
import redis
import os
import time
import random
from unittest.mock import patch, MagicMock
import warnings

from expertise.service.utils import RedisDatabase, JobConfig, APIRequest, JobStatus
from expertise.service.expertise import BaseExpertiseService


class TestRedisSearch:
    """
    Integration tests for the Redis search functionality.
    
    These tests verify the full functionality of retrieving and filtering jobs,
    including the sorted results.
    """
    
    @pytest.fixture
    def redis_db(self):
        """Redis database connection fixture"""
        db = RedisDatabase(
            host='localhost',
            port=6379,
            db=10,
            sync_on_disk=False  # Don't require disk synchronization for tests
        )
        # Clean up any test keys before the test
        keys = db.db.keys("job:test_*")
        if keys:
            db.db.delete(*keys)
        keys = db.db.keys("user:test_*")
        if keys:
            db.db.delete(*keys)
            
        yield db
        
        # Clean up test keys after the test
        keys = db.db.keys("job:test_*")
        if keys:
            db.db.delete(*keys)
        keys = db.db.keys("user:test_*")
        if keys:
            db.db.delete(*keys)
    
    @pytest.fixture
    def expertise_service(self):
        """Create a minimal expertise service for testing"""
        config = {
            'REDIS_ADDR': 'localhost',
            'REDIS_PORT': 6379,
            'REDIS_CONFIG_DB': 10,
            'WORKING_DIR': '/tmp/test_redis_search',
            'DEFAULT_CONFIG': {}
        }
        
        # Create a mock logger
        mock_logger = MagicMock()
        
        # Create the service with containerized=False to use Redis
        service = BaseExpertiseService(
            config=config,
            logger=mock_logger,
            containerized=False,
            sync_on_disk=False,
            worker_autorun=False  # Don't start the worker
        )
        
        return service
    
    @pytest.fixture
    def sample_jobs(self, redis_db):
        """Create sample jobs for the search tests"""
        # Create test jobs with various properties
        user_id = "test_search_user"
        conf_id = "TestConf/2023"
        base_time = int(time.time() * 1000)
        jobs = []
        
        # Job 1: Group-Note, Initialized
        job1 = JobConfig(
            name="Group-Note Search",
            user_id=user_id,
            job_id="test_search_1",
            job_dir="/tmp/test_search_1",
            cdate=base_time,
            mdate=base_time,
            status=JobStatus.INITIALIZED.value,
            description="Test Group-Note job",
            match_group=[f"{conf_id}/Reviewers"],
            paper_invitation=f"{conf_id}/-/Submission"
        )
        # Create API request
        job1.api_request = APIRequest({
            "name": "Group-Note Search",
            "entityA": {
                "type": "Group",
                "memberOf": f"{conf_id}/Reviewers"
            },
            "entityB": {
                "type": "Note",
                "invitation": f"{conf_id}/-/Submission"
            }
        })
        jobs.append(job1)
        
        # Job 2: Group-Group, Completed
        job2 = JobConfig(
            name="Group-Group Search",
            user_id=user_id,
            job_id="test_search_2",
            job_dir="/tmp/test_search_2",
            cdate=base_time - 60000,  # 1 minute older
            mdate=base_time,
            status=JobStatus.COMPLETED.value,
            description="Test Group-Group job",
            match_group=[f"{conf_id}/Reviewers"],
            alternate_match_group=[f"{conf_id}/Authors"]
        )
        # Create API request
        job2.api_request = APIRequest({
            "name": "Group-Group Search",
            "entityA": {
                "type": "Group",
                "memberOf": f"{conf_id}/Reviewers"
            },
            "entityB": {
                "type": "Group",
                "memberOf": f"{conf_id}/Authors"
            }
        })
        jobs.append(job2)
        
        # Job 3: Note-Note, Error
        job3 = JobConfig(
            name="Note-Note Search",
            user_id=user_id,
            job_id="test_search_3",
            job_dir="/tmp/test_search_3",
            cdate=base_time - 120000,  # 2 minutes older
            mdate=base_time,
            status=JobStatus.ERROR.value,
            description="Test Note-Note job",
            paper_id="123"
        )
        # Create API request
        job3.api_request = APIRequest({
            "name": "Note-Note Search",
            "entityA": {
                "type": "Note",
                "id": "123"
            },
            "entityB": {
                "type": "Note",
                "id": "456"
            }
        })
        jobs.append(job3)
        
        # Job 4: Different user
        job4 = JobConfig(
            name="Different User",
            user_id="another_user",
            job_id="test_search_4",
            job_dir="/tmp/test_search_4",
            cdate=base_time,
            mdate=base_time,
            status=JobStatus.INITIALIZED.value,
            description="Test job for another user",
            match_group=[f"{conf_id}/Reviewers"]
        )
        # Create API request
        job4.api_request = APIRequest({
            "name": "Different User",
            "entityA": {
                "type": "Group",
                "memberOf": f"{conf_id}/Reviewers"
            },
            "entityB": {
                "type": "Note",
                "invitation": f"{conf_id}/-/Submission"
            }
        })
        jobs.append(job4)
        
        # Save all jobs to Redis
        for job in jobs:
            with patch('os.path.isdir', return_value=True):  # Mock directory existence check
                redis_db.save_job(job)
        
        return jobs
    
    def test_load_all_jobs_sorting(self, redis_db, sample_jobs):
        """Test that load_all_jobs returns jobs sorted by creation date"""
        # Get the user ID from the first job
        user_id = sample_jobs[0].user_id
        
        # Load all jobs for this user
        with patch('os.path.isdir', return_value=True):  # Mock directory existence check
            configs = redis_db.load_all_jobs(user_id)
        
        # Verify jobs are loaded
        assert len(configs) == 3  # Should be 3 jobs for this user
        
        # Verify sorting - should be newest first
        assert configs[0].job_id == "test_search_1"
        assert configs[1].job_id == "test_search_2"
        assert configs[2].job_id == "test_search_3"
        
        # Verify the order by checking creation dates
        for i in range(len(configs) - 1):
            assert configs[i].cdate > configs[i + 1].cdate
    
    def test_filter_by_status(self, expertise_service, sample_jobs):
        """Test filtering jobs by status"""
        # Mock the service's redis to use our test redis
        user_id = sample_jobs[0].user_id
        
        # Set up the query parameters for filtering by status
        query_params = {"status": JobStatus.INITIALIZED.value}
        
        # Mock the isdir check to always return True for test jobs
        with patch('os.path.isdir', return_value=True):
            # Get all statuses filtered by initialized status
            result = expertise_service.get_expertise_all_status(user_id, query_params)
        
        # Verify results
        assert len(result["results"]) == 1
        assert result["results"][0]["status"] == JobStatus.INITIALIZED.value
        assert result["results"][0]["jobId"] == "test_search_1"
    
    def test_filter_by_member_of(self, expertise_service, sample_jobs):
        """Test filtering jobs by memberOf field"""
        # Mock the service's redis to use our test redis
        user_id = sample_jobs[0].user_id
        
        # Set up the query parameters for filtering by memberOf
        conf_id = "TestConf/2023"
        query_params = {"entityA.memberOf": f"{conf_id}/Reviewers"}
        
        # Mock the isdir check to always return True for test jobs
        with patch('os.path.isdir', return_value=True):
            # Get all statuses filtered by memberOf
            result = expertise_service.get_expertise_all_status(user_id, query_params)
        
        # Verify results - should include both Group-Note and Group-Group
        assert len(result["results"]) == 2
        # Sort by job ID to make the test deterministic
        result["results"].sort(key=lambda x: x["jobId"])
        assert result["results"][0]["jobId"] == "test_search_1"
        assert result["results"][1]["jobId"] == "test_search_2"
    
    def test_filter_by_invitation(self, expertise_service, sample_jobs):
        """Test filtering jobs by invitation field"""
        # Mock the service's redis to use our test redis
        user_id = sample_jobs[0].user_id
        
        # Set up the query parameters for filtering by invitation
        conf_id = "TestConf/2023"
        query_params = {"entityB.invitation": f"{conf_id}/-/Submission"}
        
        # Mock the isdir check to always return True for test jobs
        with patch('os.path.isdir', return_value=True):
            # Get all statuses filtered by invitation
            result = expertise_service.get_expertise_all_status(user_id, query_params)
        
        # Verify results - should include only Group-Note
        assert len(result["results"]) == 1
        assert result["results"][0]["jobId"] == "test_search_1"
    
    def test_combined_filters(self, expertise_service, sample_jobs):
        """Test combining multiple filters"""
        # Mock the service's redis to use our test redis
        user_id = sample_jobs[0].user_id
        
        # Set up the query parameters for multiple filters
        conf_id = "TestConf/2023"
        query_params = {
            "status": JobStatus.INITIALIZED.value,
            "entityA.memberOf": f"{conf_id}/Reviewers"
        }
        
        # Mock the isdir check to always return True for test jobs
        with patch('os.path.isdir', return_value=True):
            # Get all statuses with combined filters
            result = expertise_service.get_expertise_all_status(user_id, query_params)
        
        # Verify results - should include only the specific job that matches all filters
        assert len(result["results"]) == 1
        assert result["results"][0]["jobId"] == "test_search_1"
        assert result["results"][0]["status"] == JobStatus.INITIALIZED.value
    
    def test_superuser_access(self, expertise_service, sample_jobs):
        """Test that superusers can access all jobs"""
        from expertise.service.utils import SUPERUSER_IDS
        
        # Use a superuser ID
        superuser_id = SUPERUSER_IDS[0]
        
        # Mock the isdir check to always return True for test jobs
        with patch('os.path.isdir', return_value=True):
            # Get all statuses as superuser
            result = expertise_service.get_expertise_all_status(superuser_id, {})
        
        # Verify results - should include all jobs including the one from another user
        assert len(result["results"]) >= 4  # At least our 4 sample jobs
        
        # Check that all our sample jobs are included
        job_ids = {job["jobId"] for job in result["results"]}
        for job in sample_jobs:
            assert job.job_id in job_ids


if __name__ == "__main__":
    pytest.main()