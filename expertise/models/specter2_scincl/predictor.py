from tqdm import tqdm
import json
import os

class Predictor:
    def _build_embedding_jsonl(self, paper, embedding):
        data = {
            'paper_id': paper['paper_id'],
            'embedding': embedding.detach().cpu().numpy().tolist()
        }
        if 'weight' in paper:
            data['weight'] = paper['weight']
        return json.dumps(data) + '\n'

    def _load_cached_publication_embeddings(self, publications_path):
        """Return {paper_id: embedding} from cached_<basename> next to publications_path.

        The worker writes this file alongside the dataset before tarball upload;
        Vertex extracts it next to where publications_path will be written.
        """
        if not publications_path:
            return {}
        publications_path = str(publications_path)
        cache_path = os.path.join(
            os.path.dirname(publications_path),
            f"cached_{os.path.basename(publications_path)}",
        )
        if not os.path.exists(cache_path):
            return {}
        lookup = {}
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
        return lookup

    def _build_cached_embedding_jsonl(self, paper, embedding):
        data = {'paper_id': paper['paper_id'], 'embedding': embedding}
        if 'weight' in paper:
            data['weight'] = paper['weight']
        return json.dumps(data) + '\n'
