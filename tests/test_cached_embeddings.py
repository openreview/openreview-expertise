"""
Tests for the embeddings cache functionality

This test suite verifies that the EmbeddingsCache class correctly:
- Computes content hashes for deduplication
- Saves and retrieves embeddings from MongoDB
- Determines cache hits vs misses based on content changes
- Integrates properly with model predictors

Note: These tests require a MongoDB instance running on localhost:27017
"""

import pytest
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from expertise.embeddings_cache import EmbeddingsCache

# MongoDB connection settings for tests
MONGODB_URI = "mongodb://localhost:27017"
TEST_DB_NAME = "openreview_test"
TEST_COLLECTION_NAME = "openreview_embeddings"


@pytest.fixture(scope="session")
def check_mongodb():
    """Check if MongoDB is available, skip tests if not"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        client.close()
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError):
        pytest.skip("MongoDB not available at localhost:27017")


@pytest.fixture(scope="function")
def clean_cache(check_mongodb):
    """Provide a clean cache instance and clean up after each test"""
    cache = EmbeddingsCache(
        mongodb_uri=MONGODB_URI,
        db_name=TEST_DB_NAME,
        collection_name=TEST_COLLECTION_NAME
    )

    # Connect to cache
    cache.connect()

    # Clean up any existing test data
    if cache.is_connected:
        cache.collection.delete_many({})

    yield cache

    # Clean up after test
    if cache.is_connected:
        cache.collection.delete_many({})
        cache.close()


@pytest.fixture(scope="function")
def disconnected_cache(check_mongodb):
    """Provide a cache instance that is intentionally not connected"""
    cache = EmbeddingsCache(
        mongodb_uri=MONGODB_URI,
        db_name=TEST_DB_NAME,
        collection_name=TEST_COLLECTION_NAME
    )
    # Don't connect - leave it disconnected for testing
    yield cache


class TestContentHashing:
    """Test the content hash computation functionality"""

    def test_compute_content_hash_basic(self):
        """Test that compute_content_hash produces consistent hashes"""
        title = "Neural Networks for Machine Learning"
        abstract = "This paper presents a novel approach to deep learning."

        # Compute hash twice - should be identical
        hash1 = EmbeddingsCache.compute_content_hash(title, abstract)
        hash2 = EmbeddingsCache.compute_content_hash(title, abstract)

        assert hash1 == hash2, "Same content should produce same hash"
        assert isinstance(hash1, str), "Hash should be a string"
        assert len(hash1) > 0, "Hash should not be empty"

    def test_compute_content_hash_different_content(self):
        """Test that different content produces different hashes"""
        title1 = "Neural Networks for Machine Learning"
        abstract1 = "This paper presents a novel approach to deep learning."

        title2 = "Different Title"
        abstract2 = "Different abstract content."

        hash1 = EmbeddingsCache.compute_content_hash(title1, abstract1)
        hash2 = EmbeddingsCache.compute_content_hash(title2, abstract2)

        assert hash1 != hash2, "Different content should produce different hashes"

    def test_compute_content_hash_title_change(self):
        """Test that changing only the title produces a different hash"""
        title1 = "Original Title"
        title2 = "Modified Title"
        abstract = "Same abstract content"

        hash1 = EmbeddingsCache.compute_content_hash(title1, abstract)
        hash2 = EmbeddingsCache.compute_content_hash(title2, abstract)

        assert hash1 != hash2, "Changing title should change hash"

    def test_compute_content_hash_abstract_change(self):
        """Test that changing only the abstract produces a different hash"""
        title = "Same title"
        abstract1 = "Original abstract content"
        abstract2 = "Modified abstract content"

        hash1 = EmbeddingsCache.compute_content_hash(title, abstract1)
        hash2 = EmbeddingsCache.compute_content_hash(title, abstract2)

        assert hash1 != hash2, "Changing abstract should change hash"

    def test_compute_content_hash_empty_strings(self):
        """Test that empty strings produce a valid hash"""
        hash1 = EmbeddingsCache.compute_content_hash("", "")
        assert isinstance(hash1, str), "Empty strings should produce a valid hash"
        assert len(hash1) > 0, "Hash of empty strings should not be empty"

    def test_compute_content_hash_unicode(self):
        """Test that unicode characters are handled correctly"""
        title = "机器学习神经网络"
        abstract = "これは深層学習の新しいアプローチです"

        hash1 = EmbeddingsCache.compute_content_hash(title, abstract)
        hash2 = EmbeddingsCache.compute_content_hash(title, abstract)

        assert hash1 == hash2, "Unicode content should produce consistent hashes"


class TestEmbeddingsCacheBasic:
    """Test basic cache operations"""

    def test_cache_initialization(self, check_mongodb):
        """Test that cache initializes correctly"""
        cache = EmbeddingsCache(
            mongodb_uri=MONGODB_URI,
            db_name=TEST_DB_NAME,
            collection_name=TEST_COLLECTION_NAME
        )

        assert cache.mongodb_uri == MONGODB_URI
        assert cache.db_name == TEST_DB_NAME
        assert cache.collection_name == TEST_COLLECTION_NAME
        assert cache.is_connected == False

    def test_cache_connection_success(self, check_mongodb):
        """Test successful connection to MongoDB"""
        cache = EmbeddingsCache(
            mongodb_uri=MONGODB_URI,
            db_name=TEST_DB_NAME,
            collection_name=TEST_COLLECTION_NAME
        )

        result = cache.connect()

        assert result == True, "Connection should succeed"
        assert cache.is_connected == True
        assert cache.client is not None
        assert cache.db is not None
        assert cache.collection is not None

        cache.close()

    def test_cache_connection_failure(self, check_mongodb):
        """Test connection failure handling"""
        # Use invalid URI to trigger connection failure
        cache = EmbeddingsCache(
            mongodb_uri="mongodb://invalid-host:99999",
            db_name=TEST_DB_NAME,
            collection_name=TEST_COLLECTION_NAME
        )

        result = cache.connect()

        assert result == False, "Connection should fail"
        assert cache.is_connected == False


class TestSaveEmbeddings:
    """Test saving embeddings to cache"""

    def test_save_new_embedding(self, clean_cache):
        """Test saving a new embedding"""
        # Create test data
        note_id = "test_note_123"
        embedding = [0.1, 0.2, 0.3]
        content_hash = "abc123hash"

        embeddings_data = [(note_id, embedding, content_hash)]

        # Save the embedding
        result = clean_cache.save_embeddings(embeddings_data, "specter2")

        assert result == True, "Save should succeed"

        # Verify it was saved by retrieving it
        saved_docs = clean_cache.get_embeddings([note_id], "specter2")
        assert len(saved_docs) == 1, "Should have saved one document"
        assert saved_docs[0]["noteId"] == note_id
        assert saved_docs[0]["embedding"] == embedding
        assert saved_docs[0]["contentHash"] == content_hash
        assert saved_docs[0]["model"] == "specter2"

    def test_save_update_existing_with_new_hash(self, clean_cache):
        """Test updating an embedding when content hash changes"""
        note_id = "test_note_123"

        # First save with old hash
        old_embedding = [0.1, 0.2, 0.3]
        old_hash = "old_hash"
        clean_cache.save_embeddings([(note_id, old_embedding, old_hash)], "specter2")

        # Update with new hash
        new_embedding = [0.4, 0.5, 0.6]
        new_hash = "new_hash_different"
        embeddings_data = [(note_id, new_embedding, new_hash)]

        result = clean_cache.save_embeddings(embeddings_data, "specter2")

        assert result == True, "Update should succeed"

        # Verify the embedding was updated
        saved_docs = clean_cache.get_embeddings([note_id], "specter2")
        assert len(saved_docs) == 1
        assert saved_docs[0]["embedding"] == new_embedding, "Embedding should be updated"
        assert saved_docs[0]["contentHash"] == new_hash, "Hash should be updated"

    def test_skip_update_with_same_hash(self, clean_cache):
        """Test that embeddings with same hash are not updated"""
        note_id = "test_note_123"
        same_hash = "same_hash_value"
        embedding = [0.1, 0.2, 0.3]

        # First save
        clean_cache.save_embeddings([(note_id, embedding, same_hash)], "specter2")

        # Get the mdate from first save
        first_save = clean_cache.get_embeddings([note_id], "specter2")[0]
        first_mdate = first_save["mdate"]

        # Wait a bit to ensure time would change
        time.sleep(0.1)

        # Try to save again with same hash
        result = clean_cache.save_embeddings([(note_id, embedding, same_hash)], "specter2")

        assert result == True, "Should return True even when skipping"

        # Verify mdate didn't change (confirming no update happened)
        second_save = clean_cache.get_embeddings([note_id], "specter2")[0]
        assert second_save["mdate"] == first_mdate, "mdate should not change when skipping update"

    def test_mdate_updated_on_save(self, clean_cache):
        """Test that mdate is set to current time when saving"""
        note_id = "test_note_123"
        embedding = [0.1, 0.2, 0.3]
        content_hash = "hash123"

        # Get approximate current time
        time_before = int(time.time())

        # Save
        clean_cache.save_embeddings([(note_id, embedding, content_hash)], "specter2")

        time_after = int(time.time())

        # Retrieve and verify mdate
        saved_docs = clean_cache.get_embeddings([note_id], "specter2")
        saved_mdate = saved_docs[0]["mdate"]

        assert time_before <= saved_mdate <= time_after, "mdate should be current time"


class TestGetBatchCacheInfo:
    """Test batch cache info retrieval"""

    def test_all_cached(self, clean_cache):
        """Test when all items are in cache with matching hashes"""
        # Prepare batch data
        batch_data = [
            ("note_1", {"title": "Title 1", "abstract": "Abstract 1"}),
            ("note_2", {"title": "Title 2", "abstract": "Abstract 2"})
        ]

        # Compute expected hashes and save to cache
        hash1 = EmbeddingsCache.compute_content_hash("Title 1", "Abstract 1")
        hash2 = EmbeddingsCache.compute_content_hash("Title 2", "Abstract 2")

        clean_cache.save_embeddings([
            ("note_1", [0.1, 0.2], hash1),
            ("note_2", [0.3, 0.4], hash2)
        ], "specter2")

        # Get cache info
        cached_items, uncached_items = clean_cache.get_batch_cache_info(batch_data, "specter2")

        assert len(cached_items) == 2, "Both items should be cached"
        assert len(uncached_items) == 0, "No items should be uncached"

    def test_all_uncached(self, clean_cache):
        """Test when no items are in cache"""
        batch_data = [
            ("note_1", {"title": "Title 1", "abstract": "Abstract 1"}),
            ("note_2", {"title": "Title 2", "abstract": "Abstract 2"})
        ]

        # Get cache info (cache is empty)
        cached_items, uncached_items = clean_cache.get_batch_cache_info(batch_data, "specter2")

        assert len(cached_items) == 0, "No items should be cached"
        assert len(uncached_items) == 2, "Both items should be uncached"

        # Verify uncached items have content_hash
        for idx, note_id, paper_data, content_hash in uncached_items:
            assert content_hash is not None, "Uncached item should have content_hash"
            assert isinstance(content_hash, str), "content_hash should be string"

    def test_mixed_cached_uncached(self, clean_cache):
        """Test when some items are cached and some are not"""
        batch_data = [
            ("note_1", {"title": "Title 1", "abstract": "Abstract 1"}),
            ("note_2", {"title": "Title 2", "abstract": "Abstract 2"}),
            ("note_3", {"title": "Title 3", "abstract": "Abstract 3"})
        ]

        # Only save note_1 to cache
        hash1 = EmbeddingsCache.compute_content_hash("Title 1", "Abstract 1")
        clean_cache.save_embeddings([("note_1", [0.1, 0.2], hash1)], "specter2")

        # Get cache info
        cached_items, uncached_items = clean_cache.get_batch_cache_info(batch_data, "specter2")

        assert len(cached_items) == 1, "One item should be cached"
        assert len(uncached_items) == 2, "Two items should be uncached"
        assert cached_items[0][1] == "note_1", "note_1 should be cached"

    def test_content_changed_hash_mismatch(self, clean_cache):
        """Test that items with changed content (different hash) are treated as uncached"""
        # Save with OLD content
        old_hash = EmbeddingsCache.compute_content_hash("Old Title", "Old Abstract")
        clean_cache.save_embeddings([("note_1", [0.1, 0.2], old_hash)], "specter2")

        # Query with NEW content (different hash)
        batch_data = [
            ("note_1", {"title": "New Title", "abstract": "New Abstract"})
        ]

        # Get cache info
        cached_items, uncached_items = clean_cache.get_batch_cache_info(batch_data, "specter2")

        assert len(cached_items) == 0, "Item should NOT be cached (hash mismatch)"
        assert len(uncached_items) == 1, "Item should be uncached due to content change"

    def test_disconnected_cache(self, disconnected_cache):
        """Test behavior when cache is disconnected"""
        batch_data = [
            ("note_1", {"title": "Title 1", "abstract": "Abstract 1"}),
            ("note_2", {"title": "Title 2", "abstract": "Abstract 2"})
        ]

        # Get cache info
        cached_items, uncached_items = disconnected_cache.get_batch_cache_info(batch_data, "specter2")

        assert len(cached_items) == 0, "No items should be cached when disconnected"
        assert len(uncached_items) == 2, "All items should be uncached when disconnected"

        # Verify all uncached items still have content_hash computed
        for item in uncached_items:
            assert len(item) == 4, "Uncached item should have 4 elements"
            assert item[3] is not None, "Should have content_hash even when disconnected"


class TestCacheBuffering:
    """Test the EmbeddingsCache buffering functionality"""

    def test_buffer_initialization(self, check_mongodb):
        """Test that cache initializes with buffer correctly"""
        cache = EmbeddingsCache(
            mongodb_uri=MONGODB_URI,
            db_name=TEST_DB_NAME,
            collection_name=TEST_COLLECTION_NAME,
            buffer_flush_size=50
        )

        assert cache.embeddings_buffer == {}
        assert cache.buffer_flush_size == 50

    def test_buffer_accumulation(self, clean_cache):
        """Test that embeddings accumulate in buffer before flushing"""
        # Configure with large buffer so it doesn't auto-flush
        clean_cache.buffer_flush_size = 100

        # Add 5 items - should accumulate in buffer
        embeddings_data = []
        for i in range(5):
            note_id = f"test_note_{i}"
            embedding = [0.1 * i, 0.2 * i, 0.3 * i]
            content_hash = f"hash_{i}"
            embeddings_data.append((note_id, embedding, content_hash))

        # Save batch - should accumulate in buffer
        result = clean_cache.save_batch_embeddings(embeddings_data, "test_model")
        assert result == True

        # Verify buffer has items
        assert "test_model" in clean_cache.embeddings_buffer
        assert len(clean_cache.embeddings_buffer["test_model"]) == 5

        # Verify items are NOT yet in MongoDB
        note_ids = [f"test_note_{i}" for i in range(5)]
        saved_docs = clean_cache.get_embeddings(note_ids, "test_model")
        assert len(saved_docs) == 0

    def test_auto_flush_on_threshold(self, clean_cache):
        """Test that buffer auto-flushes when threshold is reached"""
        # Configure with small buffer for testing
        clean_cache.buffer_flush_size = 3

        # Add 3 items - should trigger auto-flush
        embeddings_data = []
        for i in range(3):
            note_id = f"test_note_{i}"
            embedding = [0.1 * i, 0.2 * i, 0.3 * i]
            content_hash = f"hash_{i}"
            embeddings_data.append((note_id, embedding, content_hash))

        # Save batch - should trigger flush
        result = clean_cache.save_batch_embeddings(embeddings_data, "test_model")
        assert result == True

        # Verify buffer is now empty
        assert len(clean_cache.embeddings_buffer.get("test_model", [])) == 0

        # Verify items ARE in MongoDB
        note_ids = [f"test_note_{i}" for i in range(3)]
        saved_docs = clean_cache.get_embeddings(note_ids, "test_model")
        assert len(saved_docs) == 3

    def test_manual_flush_with_force(self, clean_cache):
        """Test that force flush works regardless of buffer size"""
        # Configure with large buffer
        clean_cache.buffer_flush_size = 100

        # Add only 3 items (below threshold)
        embeddings_data = []
        for i in range(3):
            note_id = f"test_note_{i}"
            embedding = [0.1 * i, 0.2 * i, 0.3 * i]
            content_hash = f"hash_{i}"
            embeddings_data.append((note_id, embedding, content_hash))

        # Save batch - should accumulate only
        clean_cache.save_batch_embeddings(embeddings_data, "test_model")

        # Verify buffer has items
        assert len(clean_cache.embeddings_buffer["test_model"]) == 3

        # Force flush
        result = clean_cache.flush_buffer(model="test_model", force=True)
        assert result == True

        # Verify buffer is now empty
        assert len(clean_cache.embeddings_buffer.get("test_model", [])) == 0

        # Verify items ARE in MongoDB
        note_ids = [f"test_note_{i}" for i in range(3)]
        saved_docs = clean_cache.get_embeddings(note_ids, "test_model")
        assert len(saved_docs) == 3

    def test_flush_without_force_below_threshold(self, clean_cache):
        """Test that flush without force doesn't flush if below threshold"""
        # Configure with large buffer
        clean_cache.buffer_flush_size = 100

        # Add only 3 items
        embeddings_data = []
        for i in range(3):
            note_id = f"test_note_{i}"
            embedding = [0.1 * i, 0.2 * i, 0.3 * i]
            content_hash = f"hash_{i}"
            embeddings_data.append((note_id, embedding, content_hash))

        # Save batch
        clean_cache.save_batch_embeddings(embeddings_data, "test_model")

        # Try to flush without force
        result = clean_cache.flush_buffer(model="test_model", force=False)
        assert result == True

        # Buffer should still have items (not flushed)
        assert len(clean_cache.embeddings_buffer["test_model"]) == 3

        # Items should NOT be in MongoDB
        note_ids = [f"test_note_{i}" for i in range(3)]
        saved_docs = clean_cache.get_embeddings(note_ids, "test_model")
        assert len(saved_docs) == 0

    def test_multiple_models_buffering(self, clean_cache):
        """Test that different models have separate buffers"""
        clean_cache.buffer_flush_size = 100

        # Add items for model1
        embeddings_data_1 = [("note_1a", [0.1, 0.2], "hash1a"), ("note_1b", [0.3, 0.4], "hash1b")]
        clean_cache.save_batch_embeddings(embeddings_data_1, "model1")

        # Add items for model2
        embeddings_data_2 = [("note_2a", [0.5, 0.6], "hash2a"), ("note_2b", [0.7, 0.8], "hash2b")]
        clean_cache.save_batch_embeddings(embeddings_data_2, "model2")

        # Verify both models have separate buffers
        assert len(clean_cache.embeddings_buffer["model1"]) == 2
        assert len(clean_cache.embeddings_buffer["model2"]) == 2

        # Flush only model1
        clean_cache.flush_buffer(model="model1", force=True)

        # model1 buffer should be empty, model2 should still have items
        assert len(clean_cache.embeddings_buffer.get("model1", [])) == 0
        assert len(clean_cache.embeddings_buffer["model2"]) == 2

        # Verify only model1 items are in database
        assert len(clean_cache.get_embeddings(["note_1a", "note_1b"], "model1")) == 2
        assert len(clean_cache.get_embeddings(["note_2a", "note_2b"], "model2")) == 0

    def test_flush_all_models(self, clean_cache):
        """Test flushing all models at once"""
        clean_cache.buffer_flush_size = 100

        # Add items for multiple models
        clean_cache.save_batch_embeddings([("note_1", [0.1], "hash1")], "model1")
        clean_cache.save_batch_embeddings([("note_2", [0.2], "hash2")], "model2")

        # Flush all models
        result = clean_cache.flush_buffer(model=None, force=True)
        assert result == True

        # All buffers should be empty
        assert len(clean_cache.embeddings_buffer.get("model1", [])) == 0
        assert len(clean_cache.embeddings_buffer.get("model2", [])) == 0

        # All items should be in database
        assert len(clean_cache.get_embeddings(["note_1"], "model1")) == 1
        assert len(clean_cache.get_embeddings(["note_2"], "model2")) == 1

    def test_flush_empty_buffer(self, clean_cache):
        """Test that flushing empty buffer works without error"""
        result = clean_cache.flush_buffer(model="test_model", force=True)
        assert result == True

    def test_buffer_with_disconnected_cache(self, disconnected_cache):
        """Test that buffering handles disconnected cache gracefully"""
        embeddings_data = [("note_1", [0.1, 0.2], "hash1")]

        # Should return False since not connected
        result = disconnected_cache.save_batch_embeddings(embeddings_data, "test_model")
        assert result == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ===================================================================================
# SETUP INSTRUCTIONS
# ===================================================================================
#
# These tests require a MongoDB instance running locally:
#
# 1. Install MongoDB (if not already installed):
#    - macOS: brew install mongodb-community
#    - Ubuntu: sudo apt-get install mongodb
#    - Or use Docker: docker run -d -p 27017:27017 mongo
#
# 2. Start MongoDB (if not already running):
#    - macOS: brew services start mongodb-community
#    - Ubuntu: sudo systemctl start mongodb
#    - Docker: docker start <container-id>
#
# 3. Verify MongoDB is running:
#    mongo --eval "db.adminCommand('ping')"
#
# 4. Run the tests:
#    pytest tests/test_cached_embeddings.py -v
#
# Note: Tests will be skipped automatically if MongoDB is not available.
# The tests use a dedicated test database (openreview_test) which is
# cleaned up after each test to avoid interference.
#
# ===================================================================================
