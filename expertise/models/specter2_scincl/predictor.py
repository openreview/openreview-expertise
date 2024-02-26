from tqdm import tqdm
import openreview
import itertools
import torch, json
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

    def _fetch_batches(self, dict_data, batch_size):
        iterator = iter(dict_data.items())
        for _ in itertools.count():
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    def _batch_predict(self, batch_data):
        jsonl_out = []
        text_batch = [d[1]['title'] + self.tokenizer.sep_token + (d[1].get('abstract') or '') for d in batch_data]
        # preprocess the input
        inputs = self.tokenizer(text_batch, padding=True, truncation=True,
                                        return_tensors="pt", return_token_type_ids=False, max_length=512)
        inputs = inputs.to(self.cuda_device)
        with torch.no_grad():
            output = self.model(**inputs)
        # take the first token in the batch as the embedding
        embeddings = output.last_hidden_state[:, 0, :]

        for paper, embedding in zip(batch_data, embeddings):
            paper = paper[1]
            jsonl_out.append({'paper_id': paper['paper_id'], 'embedding': embedding})

        # clean up batch data
        del embeddings
        del output
        del inputs
        torch.cuda.empty_cache()
        return jsonl_out

    def _create_embeddings(self, metadata_file, embedding_path):
        with open(metadata_file, 'r') as f:
            paper_data = json.load(f)

        emb_jsonl = []
        for batch_data in tqdm(self._fetch_batches(paper_data, self.batch_size), desc='Embedding Subs', total=int(len(paper_data.keys())/self.batch_size), unit="batches"):
            emb_jsonl.extend(self._batch_predict(batch_data))

        torch.save(emb_jsonl, embedding_path)

    def _load_emb_file(emb_path, cuda_device):
        loaded_embeddings = torch.load(emb_path)
        paper_emb_size_default = 768
        id_list = []
        emb_list = []
        bad_id_set = set()
        for line in loaded_embeddings:
            paper_data = line
            paper_id = paper_data['paper_id']
            paper_emb_size = len(paper_data['embedding'])
            assert paper_emb_size == 0 or paper_emb_size == paper_emb_size_default
            if paper_emb_size == 0:
                paper_emb = [0] * paper_emb_size_default
                bad_id_set.add(paper_id)
            else:
                paper_emb = paper_data['embedding']
            id_list.append(paper_id)
            emb_list.append(paper_emb)
        if len(emb_list) == 0:
            raise openreview.OpenReviewException('No embeddings found. Please check that you have at least 1 submission submitted and that you have run the Post Submission stage.')
        emb_tensor = torch.stack(emb_list)
        emb_tensor = emb_tensor / (emb_tensor.norm(dim=1, keepdim=True) + 0.000000000001)
        print(len(bad_id_set))
        return emb_tensor, id_list, bad_id_set