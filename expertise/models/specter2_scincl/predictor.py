from tqdm import tqdm
import json
import torch
from expertise.embeddings_cache import EmbeddingsCache

class Predictor:
    def _build_embedding_jsonl(self, paper, embedding):
        data = {
            'paper_id': paper['paper_id'],
            'embedding': embedding.detach().cpu().numpy().tolist()
        }
        if 'weight' in paper:
            data['weight'] = paper['weight']
        return json.dumps(data) + '\n'

    def _get_batch_cache_info(self, batch_data):
        # Use cache to analyze batch and get cached/uncached items
        cached_items, uncached_items = [], []
        if self.use_cache:
            cached_items, uncached_items = self.embeddings_cache.get_batch_cache_info(batch_data, self.model_name)
        else:
            # If no cache, all items need computation
            # Compute content hash for each item even without cache for consistency
            uncached_with_hash = []
            for idx, (note_id, paper_data) in enumerate(batch_data):
                title = paper_data.get('title', '')
                abstract = paper_data.get('abstract', '')
                content_hash = EmbeddingsCache.compute_content_hash(title, abstract)
                uncached_with_hash.append((idx, note_id, paper_data, content_hash))
            uncached_items = uncached_with_hash

        return cached_items, uncached_items

    def _save_batch_embeddings(self, uncached_items, embeddings):
        if not self.use_cache:
            return True

        computed_for_cache = []
        for i, (idx, note_id, paper_data, content_hash) in enumerate(uncached_items):
            embedding = embeddings[i]

            embedding_list = embedding.detach().cpu().numpy().tolist()
            computed_for_cache.append((note_id, embedding_list, content_hash))

        # Save computed embeddings to cache
        if computed_for_cache:
            self.embeddings_cache.save_batch_embeddings(computed_for_cache, self.model_name)

    def _compute_embeddings_for_batch(self, uncached_items):
        """
        Compute embeddings for a batch of uncached items.

        This method should be implemented by child classes to handle model-specific
        tokenization and embedding computation.

        Args:
            uncached_items: List of tuples (idx, note_id, paper_data, content_hash)

        Returns:
            Tuple of (embeddings, cleanup_objects) where:
            - embeddings: Tensor of computed embeddings
            - cleanup_objects: List of objects to delete for GPU memory cleanup
        """
        raise NotImplementedError("Subclasses must implement _compute_embeddings_for_batch")

    def _batch_predict(self, batch_data):
        """
        Predict embeddings for a batch of papers, using cache when possible.

        Args:
            batch_data: List of tuples (note_id, paper_data_dict)

        Returns:
            List of JSONL strings with embeddings for each paper
        """
        jsonl_out = []

        cached_items, uncached_items = self._get_batch_cache_info(batch_data)

        # Handle cached items - create JSONL output from cached embeddings
        cached_results = []
        for idx, note_id, cached_embedding in cached_items:
            paper_data = batch_data[idx][1]  # Get original paper data for weight info
            paper = {
                'paper_id': note_id,
                'embedding': cached_embedding
            }
            if 'weight' in paper_data:
                paper['weight'] = paper_data['weight']

            cached_results.append((idx, json.dumps(paper) + '\n'))

        # Handle uncached items - compute embeddings
        computed_results = []

        if uncached_items:
            # Compute embeddings using model-specific implementation
            embeddings, cleanup_objects = self._compute_embeddings_for_batch(uncached_items)

            # Process computed embeddings
            for i, (idx, note_id, paper_data, content_hash) in enumerate(uncached_items):
                embedding = embeddings[i]
                # Create JSONL output
                embedding_jsonl = self._build_embedding_jsonl(paper_data, embedding)
                computed_results.append((idx, embedding_jsonl))

            self._save_batch_embeddings(uncached_items, embeddings)

            # Clean up GPU memory
            for obj in cleanup_objects:
                del obj
            torch.cuda.empty_cache()

        # Merge cached and computed results in original order
        all_results = cached_results + computed_results
        all_results.sort(key=lambda x: x[0])  # Sort by original batch index
        jsonl_out = [result[1] for result in all_results]

        return jsonl_out
