from tqdm import tqdm
import json

class Predictor:
    def _build_embedding_jsonl(self, paper, embedding):
        data = {
            'paper_id': paper['paper_id'],
            'embedding': embedding.detach().cpu().numpy().tolist()
        }
        if 'weight' in paper:
            data['weight'] = paper['weight']
        return json.dumps(data) + '\n'
