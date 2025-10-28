from tqdm import tqdm
import json
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
