"""
Filter JSONL files by removing documents that already have embeddings in MongoDB.

This script reads JSONL files created by expertise/create_dataset.py and removes
documents that already have precomputed embeddings stored in MongoDB, based on
document ID and model name.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Set
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
from pymongo import ReplaceOne
import time
import xxhash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingsCache:
    """Cache for existing document embeddings with buffering support."""

    def __init__(self, mongodb_uri: str, db_name: str, collection_name: str, buffer_flush_size: int = 10000):
        """
        Initialize the embeddings cache.

        Args:
            mongodb_uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name where embeddings are stored
            buffer_flush_size: Number of embeddings to accumulate before flushing to DB
        """
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.is_connected = False
        self.buffer_flush_size = buffer_flush_size
        self.embeddings_buffer = {}  # Dict[model_name, List[tuple]]
    
    def connect(self) -> bool:
        """
        Connect to MongoDB.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = MongoClient(self.mongodb_uri)
            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info(f"Connected to MongoDB: {self.db_name}.{self.collection_name}")
            self.is_connected = True
            return True
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self.is_connected = False
            return False
    
    def close(self):
        """Close MongoDB connection."""
        if self.client and self.is_connected:
            self.client.close()
            logger.info("MongoDB connection closed")

    @staticmethod
    def compute_content_hash(title: str, abstract: str) -> str:
        """
        Compute a hash of the concatenated title and abstract.
        Uses xxHash (fastest) if available, otherwise falls back to MD5.

        Args:
            title: Document title
            abstract: Document abstract

        Returns:
            Hash of the concatenated content (xxHash or MD5)
        """
        content = f"{title}{abstract}"
        content_bytes = content.encode('utf-8')

        # xxHash is much faster than cryptographic hashes
        return xxhash.xxh64(content_bytes).hexdigest()
    
    def save_embeddings(self, embeddings_data: List[tuple], model: str) -> bool:
        """
        Save multiple embeddings to MongoDB efficiently using bulk_write.
        Only updates if the contentHash differs from existing document's contentHash.

        Args:
            embeddings_data: List of tuples (note_id, embedding, content_hash)
            model: Model name used to generate the embeddings

        Returns:
            True if all saved/updated successfully, False otherwise
        """
        if not self.is_connected or not embeddings_data:
            return False

        try:
            # Fetch existing documents for noteIds and model
            note_ids = [note_id for note_id, _, _ in embeddings_data]
            existing_docs = self.get_embeddings(note_ids, model)
            existing_map = {doc["noteId"]: doc for doc in existing_docs}

            current_time = int(time.time())
            operations = []
            for note_id, embedding, content_hash in embeddings_data:
                existing_doc = existing_map.get(note_id)
                # Only update if contentHash differs
                if existing_doc and existing_doc.get("contentHash") == content_hash:
                    logger.info(f"Skipping update for note {note_id} (model: {model}) - content unchanged")
                    continue
                doc = {
                    "noteId": note_id,
                    "model": model,
                    "embedding": embedding,
                    "contentHash": content_hash,
                    "mdate": current_time
                }
                operations.append(
                    ReplaceOne(
                        {"noteId": note_id, "model": model},
                        doc,
                        upsert=True
                    )
                )

            if operations:
                result = self.collection.bulk_write(operations)
                logger.info(f"Bulk write: {result.modified_count} updated, {result.upserted_count} inserted for model '{model}'")
            else:
                logger.info("No embeddings needed to be updated/inserted in bulk write.")

            return True
        except Exception as e:
            logger.error(f"Error saving embeddings in bulk: {e}")
            return False

    def get_embeddings(self, note_ids: List[str], model: str = None) -> List[Dict]:
        """
        Get embeddings by a list of note_ids and optionally filter by model.

        Args:
            note_ids: List of Note IDs (required)
            model: Model name (optional filter)

        Returns:
            List of embedding documents matching the criteria
        """
        if not self.is_connected or not note_ids:
            return []

        try:
            # Build query
            query = {"noteId": {"$in": note_ids}}
            if model:
                query["model"] = model

            # Execute query
            cursor = self.collection.find(query)
            embeddings = list(cursor)

            logger.info(f"Found {len(embeddings)} embeddings for note_ids {note_ids}" +
                        (f" (model: {model})" if model else ""))
            return embeddings

        except Exception as e:
            logger.error(f"Error retrieving embeddings for note_ids {note_ids}: {e}")
            return []

    def get_batch_cache_info(self, batch_data: List[tuple], model: str) -> tuple:
        """
        Analyze a batch to determine which items need computation vs can use cache.

        Args:
            batch_data: List of tuples (note_id, paper_data_dict)
            model: Model name to check cache for

        Returns:
            Tuple of (cached_items, uncached_items) where:
            - cached_items: List of (index, note_id, cached_embedding)
            - uncached_items: List of (index, note_id, paper_data, content_hash)
        """
        if not self.is_connected:
            # If not connected, all items need computation
            uncached_items = []
            for idx, (note_id, paper_data) in enumerate(batch_data):
                title = paper_data.get('title', '')
                abstract = paper_data.get('abstract', '')
                content_hash = self.compute_content_hash(title, abstract)
                uncached_items.append((idx, note_id, paper_data, content_hash))
            return [], uncached_items

        cached_items = []
        uncached_items = []

        # Directly create note_ids list from batch_data
        note_ids = [note_id for note_id, _ in batch_data]

        # Fetch all cached embeddings for these note_ids in one query
        cached_embeddings = self.get_embeddings(note_ids, model)
        cached_map = {doc['noteId']: doc for doc in cached_embeddings}

        for idx, (note_id, paper_data) in enumerate(batch_data):
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            content_hash = self.compute_content_hash(title, abstract)

            cached_emb = cached_map.get(note_id)
            if cached_emb and cached_emb.get('contentHash') == content_hash:
                cached_items.append((idx, note_id, cached_emb['embedding']))
            else:
                uncached_items.append((idx, note_id, paper_data, content_hash))

        logger.info(f"Batch analysis: {len(cached_items)} cached, {len(uncached_items)} need computation")
        return cached_items, uncached_items

    def save_batch_embeddings(self, computed_items: List[tuple], model: str) -> bool:
        """
        Accumulate embeddings to buffer and flush when threshold is reached.

        Args:
            computed_items: List of tuples (note_id, embedding_list, content_hash)
            model: Model name

        Returns:
            True if accumulation/flush succeeded, False otherwise
        """
        if not self.is_connected:
            return False

        # Initialize buffer for this model if needed
        if model not in self.embeddings_buffer:
            self.embeddings_buffer[model] = []

        # Add items to buffer
        self.embeddings_buffer[model].extend(computed_items)
        logger.info(f"Added {len(computed_items)} embeddings to buffer for model '{model}' "
                   f"(buffer size: {len(self.embeddings_buffer[model])})")

        # Auto-flush if buffer is large enough
        if len(self.embeddings_buffer[model]) >= self.buffer_flush_size:
            return self.flush_buffer(model)

        return True

    def flush_buffer(self, model: str = None, force: bool = False) -> bool:
        """
        Flush accumulated embeddings to MongoDB.

        Args:
            model: Model name to flush (if None, flushes all models)
            force: If True, flush regardless of buffer size

        Returns:
            True if flush succeeded, False otherwise
        """
        if not self.is_connected:
            return True

        # Determine which models to flush
        models_to_flush = [model] if model else list(self.embeddings_buffer.keys())

        success = True
        for model_name in models_to_flush:
            if model_name not in self.embeddings_buffer or not self.embeddings_buffer[model_name]:
                continue

            buffer_size = len(self.embeddings_buffer[model_name])

            # Only flush if forced or buffer is large enough
            if not force and buffer_size < self.buffer_flush_size:
                continue

            logger.info(f"Flushing {buffer_size} embeddings for model '{model_name}'...")

            try:
                # Use the existing save_embeddings method for actual DB write
                result = self.save_embeddings(self.embeddings_buffer[model_name], model_name)
                if result:
                    # Clear buffer after successful flush
                    self.embeddings_buffer[model_name] = []
                    logger.info(f"Successfully flushed {buffer_size} embeddings for model '{model_name}'")
                else:
                    success = False
                    logger.error(f"Failed to flush embeddings for model '{model_name}'")
            except Exception as e:
                success = False
                logger.error(f"Error flushing embeddings for model '{model_name}': {e}")

        return success
