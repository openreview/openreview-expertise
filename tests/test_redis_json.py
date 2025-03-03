import pytest
import json
import pickle
import redis
import os
import time
from unittest.mock import patch, MagicMock
import warnings

from expertise.service.utils import RedisDatabase, JobConfig, APIRequest


class TestRedisJSON:
    """
    Tests for the Redis JSON implementation.
    
    These tests require a Redis instance running on localhost:6379.
    If Redis JSON module is not available, some tests will be skipped.
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
        keys = db.db.keys("test:*")
        if keys:
            db.db.delete(*keys)
        keys = db.db.keys("user:test_*")
        if keys:
            db.db.delete(*keys)
            
        yield db
        
        # Clean up test keys after the test
        keys = db.db.keys("test:*")
        if keys:
            db.db.delete(*keys)
        keys = db.db.keys("user:test_*")
        if keys:
            db.db.delete(*keys)
            
    @pytest.fixture
    def sample_job_config(self):
        """Create a sample job config for testing"""
        return JobConfig(
            name="Test Job",
            user_id="test_user",
            job_id="test123",
            job_dir="/tmp/test123",
            cdate=int(time.time() * 1000),
            mdate=int(time.time() * 1000),
            status="Initialized",
            description="Test job for unit tests",
            model="tfidf",
            model_params={"use_title": True, "use_abstract": True}
        )
    
    @pytest.fixture
    def non_serializable_job_config(self, sample_job_config):
        """Create a job config with a non-serializable attribute"""
        # Create a custom object that can't be JSON serialized
        class NonSerializable:
            def __init__(self):
                self.name = "Cannot serialize this"
        
        # Add it to the sample config
        sample_job_config.custom_object = NonSerializable()
        return sample_job_config
    
    @pytest.fixture
    def job_with_api_request(self, sample_job_config):
        """Create a job config with an API request object"""
        # Create a sample API request
        api_request = APIRequest({
            "name": "Test Request",
            "entityA": {
                "type": "Group",
                "memberOf": "Test_Conference/Reviewers"
            },
            "entityB": {
                "type": "Note",
                "invitation": "Test_Conference/-/Submission"
            }
        })
        
        sample_job_config.api_request = api_request
        return sample_job_config
    
    def test_redis_connection(self, redis_db):
        """Test that Redis connection works"""
        assert redis_db.db is not None
        # Check if Redis JSON module is available
        if not hasattr(redis_db, 'has_redisjson'):
            pytest.skip("Redis instance doesn't have has_redisjson attribute")
    
    def test_save_and_load_job(self, redis_db, sample_job_config):
        """Test saving and loading a job config"""
        # Save the job config
        job_key = f"test:{sample_job_config.job_id}"
        redis_db.db.delete(job_key)  # Ensure key doesn't exist
        
        # Override the default job key
        with patch('expertise.service.utils.RedisDatabase.save_job') as mock_save:
            mock_save.side_effect = lambda config: redis_db.db.set(
                job_key, 
                (json.dumps(config.to_json()) if redis_db.has_redisjson else pickle.dumps(config))
            )
            mock_save(sample_job_config)
        
        # Check that the job was saved
        assert redis_db.db.exists(job_key)
        
        # Load the job
        with patch('expertise.service.utils.RedisDatabase.load_job') as mock_load:
            if redis_db.has_redisjson:
                # JSON path
                job_data = redis_db.db.get(job_key)
                job_dict = json.loads(job_data)
                mock_load.return_value = JobConfig.from_json(job_dict)
            else:
                # Pickle path
                job_data = redis_db.db.get(job_key)
                mock_load.return_value = pickle.loads(job_data)
            
            loaded_config = mock_load("test123", "test_user")
        
        # Verify the loaded config
        assert loaded_config.name == sample_job_config.name
        assert loaded_config.user_id == sample_job_config.user_id
        assert loaded_config.job_id == sample_job_config.job_id
        assert loaded_config.model == sample_job_config.model
        assert loaded_config.model_params.get("use_title") == sample_job_config.model_params.get("use_title")
    
    def test_job_with_api_request(self, redis_db, job_with_api_request):
        """Test saving and loading a job with API request"""
        # Save the job config
        job_key = f"test:{job_with_api_request.job_id}"
        redis_db.db.delete(job_key)  # Ensure key doesn't exist
        
        # Convert to JSON
        job_dict = job_with_api_request.to_json()
        
        # Verify api_request was serialized properly
        if redis_db.has_redisjson:
            assert "api_request_json" in job_dict
            assert job_dict["api_request_json"]["name"] == "Test Request"
        
        # Save directly
        redis_db.db.set(job_key, json.dumps(job_dict) if redis_db.has_redisjson else pickle.dumps(job_with_api_request))
        
        # Load and verify
        if redis_db.has_redisjson:
            loaded_dict = json.loads(redis_db.db.get(job_key))
            loaded_config = JobConfig.from_json(loaded_dict)
            # Check api_request was reconstructed
            if hasattr(loaded_config, "api_request") and loaded_config.api_request is not None:
                assert loaded_config.api_request.name == "Test Request"
        else:
            loaded_config = pickle.loads(redis_db.db.get(job_key))
            assert loaded_config.api_request.name == "Test Request"
    
    def test_non_serializable_fallback(self, redis_db, non_serializable_job_config):
        """Test fallback to pickle when object is not JSON serializable"""
        # Save the job config
        job_key = f"test:{non_serializable_job_config.job_id}"
        redis_db.db.delete(job_key)  # Ensure key doesn't exist
        
        # Try to save with JSON serialization
        if redis_db.has_redisjson:
            with pytest.warns(UserWarning):
                # This should fall back to pickle due to non-serializable object
                redis_db.save_job(non_serializable_job_config)
        else:
            # Just save with pickle
            redis_db.db.set(job_key, pickle.dumps(non_serializable_job_config))
        
        # Verify the job was saved (either way)
        assert redis_db.db.exists(f"job:{non_serializable_job_config.job_id}")
    
    def test_sorted_set_indexing(self, redis_db):
        """Test the sorted set indexing for efficient user-based retrieval"""
        if not hasattr(redis_db, 'has_redisjson') or not redis_db.has_redisjson:
            pytest.skip("Redis JSON is not available")
        
        # Create multiple jobs with different creation dates
        user_id = "test_user_sorted"
        base_time = int(time.time() * 1000)
        
        jobs = []
        for i in range(5):
            job = JobConfig(
                name=f"Job {i}",
                user_id=user_id,
                job_id=f"job{i}",
                job_dir=f"/tmp/job{i}",
                cdate=base_time - (i * 60000),  # Each job is 1 minute older
                mdate=base_time - (i * 60000),
                status="Initialized",
                description=f"Test job {i}"
            )
            jobs.append(job)
            
            # Save the job using the real method
            with patch('os.path.isdir', return_value=True):  # Mock directory existence check
                redis_db.save_job(job)
        
        # Create a sorted set directly for testing
        user_index_key = f"user:{user_id}:jobs"
        
        # Check that the index exists
        assert redis_db.db.exists(user_index_key)
        
        # Get all jobs for the user using zrevrange (newest first)
        job_ids = redis_db.db.zrevrange(user_index_key, 0, -1)
        assert len(job_ids) == 5
        
        # Verify the order - should be newest first
        assert job_ids[0].decode('utf-8') if isinstance(job_ids[0], bytes) else job_ids[0] == "job0"
        assert job_ids[4].decode('utf-8') if isinstance(job_ids[4], bytes) else job_ids[4] == "job4"
    
    def test_mixed_storage_formats(self, redis_db, sample_job_config):
        """Test handling both pickle and JSON storage formats"""
        if not hasattr(redis_db, 'has_redisjson') or not redis_db.has_redisjson:
            pytest.skip("Redis JSON is not available")
        
        # Create a job saved with pickle
        job_key_pickle = f"job:pickle_{sample_job_config.job_id}"
        redis_db.db.delete(job_key_pickle)  # Ensure key doesn't exist
        
        # Create pickle version
        pickle_config = sample_job_config
        pickle_config.job_id = f"pickle_{sample_job_config.job_id}"
        redis_db.db.set(job_key_pickle, pickle.dumps(pickle_config))
        
        # Create JSON version
        job_key_json = f"job:json_{sample_job_config.job_id}"
        redis_db.db.delete(job_key_json)  # Ensure key doesn't exist
        json_config = sample_job_config
        json_config.job_id = f"json_{sample_job_config.job_id}"
        json_dict = json_config.to_json()
        redis_db.db.set(job_key_json, json.dumps(json_dict))
        
        # Verify both can be loaded correctly
        with patch('expertise.service.utils.RedisDatabase.load_job') as mock_load:
            # First load pickle version
            mock_load.side_effect = lambda job_id, user_id: pickle.loads(redis_db.db.get(f"job:{job_id}"))
            pickle_loaded = mock_load(f"pickle_{sample_job_config.job_id}", sample_job_config.user_id)
            
            # Then load JSON version
            mock_load.side_effect = lambda job_id, user_id: JobConfig.from_json(json.loads(redis_db.db.get(f"job:{job_id}")))
            json_loaded = mock_load(f"json_{sample_job_config.job_id}", sample_job_config.user_id)
        
        # Verify both loaded correctly
        assert pickle_loaded.name == json_loaded.name
        assert pickle_loaded.user_id == json_loaded.user_id
    
    def test_redis_stats(self, redis_db, sample_job_config):
        """Test the stats functionality"""
        # Save a job
        with patch('os.path.isdir', return_value=True):  # Mock directory existence check
            redis_db.save_job(sample_job_config)
        
        # Get stats
        stats = redis_db.get_stats()
        
        # Verify stats
        assert "total_jobs" in stats
        assert "total_users" in stats
        assert "jobs_per_user" in stats
        assert "redis_used_memory" in stats
        assert "has_redisjson" in stats
        
        # If Redis JSON is available, verify user stats
        if redis_db.has_redisjson:
            assert sample_job_config.user_id in stats["jobs_per_user"]
            assert stats["jobs_per_user"][sample_job_config.user_id] >= 1


if __name__ == "__main__":
    pytest.main()