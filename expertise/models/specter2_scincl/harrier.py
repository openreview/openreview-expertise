import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .scincl import SciNCLPredictor


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class HarrierPredictor(SciNCLPredictor):
    def __init__(self, specter_dir, work_dir, average_score=False, max_score=True, batch_size=16, use_cuda=True,
                 sparse_value=None, use_redis=False, dump_p2p=False, compute_paper_paper=False, percentile_select=None, venue_specific_weights=False,
                 normalize_scores=True, harrier_hf_dir=None):
        self.harrier_hf_dir = harrier_hf_dir or os.getenv('HARRIER_HF_DIR') or 'microsoft/harrier-oss-v1-0.6b'
        super().__init__(
            specter_dir=specter_dir,
            work_dir=work_dir,
            average_score=average_score,
            max_score=max_score,
            batch_size=batch_size,
            use_cuda=use_cuda,
            sparse_value=sparse_value,
            use_redis=use_redis,
            dump_p2p=dump_p2p,
            compute_paper_paper=compute_paper_paper,
            percentile_select=percentile_select,
            venue_specific_weights=venue_specific_weights,
            normalize_scores=normalize_scores,
            scincl_hf_dir=self.harrier_hf_dir,
            embedding_dim=1024,
        )
        self.model_name = 'harrier'

    def _batch_predict(self, batch_data):
        out = []
        text_batch = [d[1]['title'] + self.tokenizer.sep_token + (d[1].get('abstract') or '') for d in batch_data]
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = inputs.to(self.cuda_device)
        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = last_token_pool(output.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        for paper, embedding in zip(batch_data, embeddings):
            paper = paper[1]
            out.append({
                'paper_id': paper['paper_id'],
                'embedding': embedding.detach().cpu().float().numpy().tolist()
            })

        del embeddings
        del output
        del inputs
        torch.cuda.empty_cache()
        return out
