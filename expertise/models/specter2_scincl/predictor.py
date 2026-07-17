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

    def embed(self, metadata_file, output_path=None):
        with open(metadata_file, 'r') as f:
            paper_data = json.load(f)

        cached = getattr(self, 'cached_embeddings', None) or {}
        embeddings = {}
        remaining = {}
        for paper_id, paper in paper_data.items():
            emb = cached.get(paper_id)
            if emb is not None:
                embeddings[paper_id] = emb
            else:
                remaining[paper_id] = paper
        if cached:
            print(f"Reusing {len(embeddings)} cached embeddings; computing {len(remaining)}.")

        for batch_data in tqdm(self._fetch_batches(remaining, self.batch_size), desc='Embedding', total=int(len(remaining.keys()) / self.batch_size), unit="batches"):
            for item in self._batch_predict(batch_data):
                embeddings[item['paper_id']] = item['embedding']

        if output_path:
            with open(output_path, 'w') as f:
                for pid, emb in embeddings.items():
                    f.write(json.dumps({'paper_id': pid, 'embedding': emb}) + '\n')
        return embeddings

    def _load_cached_publication_embeddings(self, publications_path=None, cached_publications=None):
        """Load cached embeddings and return (lookup_dict, jsonl_lines).

        Args:
            publications_path: path to the embeddings file whose sidecar
                cached_<basename> will be read from disk (legacy path mode).
            cached_publications: optional dict mapping paper_id -> embedding
                (list of floats) supplied directly in memory.

        Returns:
            lookup: dict mapping paper_id -> embedding (list of floats)
            jsonl_lines: dict mapping paper_id -> original JSON line (str), or
                None when embeddings were supplied in memory.
        """
        if cached_publications is not None:
            return cached_publications, None
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
