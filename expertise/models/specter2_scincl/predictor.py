from tqdm import tqdm
import json
import os

class Predictor:
    def _build_embedding_jsonl(self, paper, embedding):
        data = {
            'paper_id': paper['paper_id'],
            'embedding': embedding.detach().cpu().numpy().tolist()
        }
        return json.dumps(data) + '\n'

    def _load_cached_publication_embeddings(self, publications_path):
        """Load cached embeddings and return (lookup_dict, jsonl_lines).

        Returns:
            lookup: dict mapping paper_id -> embedding (list of floats)
            jsonl_lines: dict mapping paper_id -> original JSON line (str)

        The jsonl_lines are the original lines from the cache file, which can be
        written directly to avoid precision loss from re-serialization.
        """
        if not publications_path:
            return {}, {}
        publications_path = str(publications_path)
        cache_path = os.path.join(
            os.path.dirname(publications_path),
            f"cached_{os.path.basename(publications_path)}",
        )
        if not os.path.exists(cache_path):
            return {}, {}
        lookup = {}
        jsonl_lines = {}
        with open(cache_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = entry.get('paper_id')
                emb = entry.get('embedding')
                if pid and emb is not None:
                    lookup[pid] = emb
                    jsonl_lines[pid] = line + '\n'
        return lookup, jsonl_lines
