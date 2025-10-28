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
    """Cache for existing document embeddings."""
    
    def __init__(self, mongodb_uri: str, db_name: str, collection_name: str):
        """
        Initialize the embeddings cache.
        
        Args:
            mongodb_uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name where embeddings are stored
        """
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.is_connected = False
    
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
        Save multiple computed embeddings to cache using bulk_write.

        Args:
            computed_items: List of tuples (note_id, embedding_list, content_hash)
            model: Model name

        Returns:
            True if all saved successfully, False otherwise
        """
        # Directly call save_embeddings for bulk efficiency
        return self.save_embeddings(computed_items, model)
