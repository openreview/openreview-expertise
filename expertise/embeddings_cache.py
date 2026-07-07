import json
from typing import List, Dict

import pyarrow.dataset as ds
import pyarrow.compute as pc


class GlobalEmbeddingsCache:
    """
    Reads embeddings from a Hive-partitioned Parquet store on GCS.

    Expected layout:
        gs://<bucket>/<prefix>/model=<model>/year_month=<YYYY-MM>/*.parquet

    Schema columns: paper_id (string), embedding (list<float>), model, year_month
    """

    def __init__(self, bucket_name: str, cache_prefix: str = "embeddings-cache"):
        self.bucket_name = bucket_name
        self.cache_prefix = cache_prefix
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is not None:
            return self._dataset
        gcs_path = f"gs://{self.bucket_name}/{self.cache_prefix}"
        self._dataset = ds.dataset(gcs_path, partitioning="hive")
        return self._dataset

    def get_embeddings(self, paper_ids: List[str], model_name: str) -> Dict[str, List[float]]:
        """
        Retrieve embeddings for the given paper_ids from the global cache.

        Returns a dict mapping paper_id -> embedding (list of floats).
        """
        if not paper_ids:
            return {}

        unique_pids = list(set(paper_ids))

        try:
            dataset = self._get_dataset()
            table = dataset.to_table(
                columns=["paper_id", "embedding"],
                filter=(
                    (pc.field("model") == model_name)
                    & pc.field("paper_id").isin(unique_pids)
                )
            )
        except Exception:
            return {}

        result = {}
        rows = table.to_pydict()
        for pid, emb in zip(rows["paper_id"], rows["embedding"]):
            result[pid] = emb
        return result

    def write_cache_file(self, paper_ids: List[str], model_name: str, dest_path: str) -> int:
        """
        Query the global cache and write the results to a local JSONL file
        in the format expected by Predictor._load_cached_publication_embeddings.
        Returns the number of embeddings written.
        """
        embeddings = self.get_embeddings(paper_ids, model_name)
        if not embeddings:
            return 0
        with open(dest_path, 'w') as f:
            for pid, emb in embeddings.items():
                f.write(json.dumps({"paper_id": pid, "embedding": emb}) + '\n')
        return len(embeddings)
