from tqdm import tqdm
import json

class Predictor:
    def _sparse_scores_helper(self, all_scores, id_index):
        counter = 0
        # Get the first note_id or profile_id
        current_id = self.preliminary_scores[0][id_index]
        if id_index == 0:
            desc = 'Note IDs'
        else:
            desc = 'Profiles IDs'
        for note_id, profile_id, score in tqdm(self.preliminary_scores, total=len(self.preliminary_scores), desc=desc):
            if counter < self.sparse_value:
                all_scores.add((note_id, profile_id, score))
            elif (note_id, profile_id)[id_index] != current_id:
                counter = 0
                all_scores.add((note_id, profile_id, score))
                current_id = (note_id, profile_id)[id_index]
            counter += 1
        return all_scores
    
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
            uncached_items = [(idx, note_id, paper_data) for idx, (note_id, paper_data) in enumerate(batch_data)]

        print(f"Loaded {len(cached_items)} cached embeddings, need to compute {len(uncached_items)} embeddings")
        return cached_items, uncached_items
    
    def _save_batch_embeddings(self, uncached_items, embeddings):
        if not self.use_cache:
            return True

        computed_for_cache = []
        for i, (idx, note_id, paper_data) in enumerate(uncached_items):
            embedding = embeddings[i]

            embedding_list = embedding.detach().cpu().numpy().tolist()
            computed_for_cache.append((note_id, embedding_list, paper_data.get('mdate', 0)))

        # Save computed embeddings to cache
        if computed_for_cache:
            self.embeddings_cache.save_batch_embeddings(computed_for_cache, self.model_name)
