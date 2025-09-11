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
import argparse
import logging

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
    
    def get_existing_embeddings(self, model_name: str) -> Set[str]:
        """
        Get set of note IDs that already have embeddings for the given model.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Set of note IDs that have embeddings
        """
        if not self.is_connected:
            return set()
            
        try:
            # Query for documents with the specific model
            cursor = self.collection.find(
                {"model": model_name},
                {"noteId": 1, "_id": 0}
            )
            existing_ids = {doc["noteId"] for doc in cursor}
            logger.info(f"Found {len(existing_ids)} existing embeddings for model '{model_name}'")
            return existing_ids
        except Exception as e:
            logger.error(f"Error querying MongoDB: {e}")
            return set()
            
    def get_existing_embedding_data(self, model_name: str) -> Dict[str, Dict]:
        """
        Get note IDs and their embeddings for the given model.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Dictionary mapping note IDs to their complete embedding data
        """
        if not self.is_connected:
            return {}
            
        try:
            # Query for documents with the specific model
            cursor = self.collection.find(
                {"model": model_name}
            )
            embedding_data = {doc["noteId"]: doc for doc in cursor}
            logger.info(f"Retrieved {len(embedding_data)} embeddings for model '{model_name}'")
            return embedding_data
        except Exception as e:
            logger.error(f"Error querying MongoDB for embedding data: {e}")
            return {}

    def save_embedding(self, note_id: str, model: str, embedding: List[float], mdate: int) -> bool:
        """
        Save embedding to MongoDB with the specified schema.
        Only updates if the mdate is greater than existing document's mdate.
        
        Args:
            note_id: Note ID
            model: Model name used to generate the embedding
            embedding: The embedding vector
            mdate: Modification date timestamp
            
        Returns:
            True if saved/updated successfully, False otherwise
        """
        if not self.is_connected:
            return False
            
        try:
            # Check if document exists with same noteId and model
            existing_doc = self.collection.find_one({
                "noteId": note_id,
                "model": model
            })
            
            # If document exists and has newer or equal mdate, skip update
            if existing_doc and existing_doc.get("mdate", 0) >= mdate:
                logger.info(f"Skipping update for note {note_id} (model: {model}) - existing mdate is newer or equal")
                return True
            
            # Prepare document with correct schema
            doc = {
                "noteId": note_id,
                "model": model,
                "embedding": embedding,
                "mdate": mdate
            }
            
            # Update or insert document
            result = self.collection.replace_one(
                {"noteId": note_id, "model": model},
                doc,
                upsert=True
            )
            
            action = "updated" if result.matched_count > 0 else "inserted"
            logger.info(f"Successfully {action} embedding for note {note_id} (model: {model})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embedding for note {note_id}: {e}")
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
            - uncached_items: List of (index, note_id, paper_data)
        """
        if not self.is_connected:
            # If not connected, all items need computation
            uncached_items = [(idx, note_id, paper_data) for idx, (note_id, paper_data) in enumerate(batch_data)]
            return [], uncached_items
            
        cached_items = []
        uncached_items = []
        
        # Directly create note_ids list from batch_data
        note_ids = [note_id for note_id, _ in batch_data]

        # Fetch all cached embeddings for these note_ids in one query
        cached_embeddings = self.get_embeddings(note_ids, model)
        cached_map = {doc['noteId']: doc for doc in cached_embeddings}

        for idx, (note_id, paper_data) in enumerate(batch_data):
            paper_mdate = paper_data.get('mdate', 0)
            cached_emb = cached_map.get(note_id)
            if cached_emb and cached_emb.get('mdate', 0) >= paper_mdate:
                cached_items.append((idx, note_id, cached_emb['embedding']))
            else:
                uncached_items.append((idx, note_id, paper_data))
        
        logger.info(f"Batch analysis: {len(cached_items)} cached, {len(uncached_items)} need computation")
        return cached_items, uncached_items

    def save_batch_embeddings(self, computed_items: List[tuple], model: str) -> bool:
        """
        Save multiple computed embeddings to cache.
        
        Args:
            computed_items: List of tuples (note_id, embedding_list, mdate)
            model: Model name
            
        Returns:
            True if all saved successfully, False otherwise
        """
        if not self.is_connected:
            return False
            
        success_count = 0
        for note_id, embedding_list, mdate in computed_items:
            if self.save_embedding(note_id, model, embedding_list, mdate):
                success_count += 1
        
        logger.info(f"Saved {success_count}/{len(computed_items)} embeddings to cache")
        return success_count == len(computed_items)
