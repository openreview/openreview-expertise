import json
import os
import tempfile
import time
from typing import List, Dict

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
from google.cloud import storage


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

    def _log_metrics(self, job_id: str, model_name: str, paper_count: int,
                     cached_count: int, scan_time_s: float, serialize_time_s: float,
                     write_time_s: float, total_time_s: float):
        try:
            metrics_prefix = f"{self.cache_prefix}-metrics"
            blob_name = f"{metrics_prefix}/scan-metrics.csv"
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                csv_path = tmp.name
            try:
                if blob.exists():
                    blob.download_to_filename(csv_path)
                    df = pd.read_csv(csv_path)
                else:
                    df = pd.DataFrame(columns=[
                        "timestamp", "job_id", "model", "paper_count", "cached_count",
                        "scan_time_s", "serialize_time_s", "write_time_s", "total_time_s"
                    ])

                df = pd.concat([df, pd.DataFrame([{
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "job_id": job_id,
                    "model": model_name,
                    "paper_count": paper_count,
                    "cached_count": cached_count,
                    "scan_time_s": scan_time_s,
                    "serialize_time_s": serialize_time_s,
                    "write_time_s": write_time_s,
                    "total_time_s": total_time_s,
                }])], ignore_index=True)
                df.to_csv(csv_path, index=False)
                blob.upload_from_filename(csv_path)
            finally:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
        except Exception as e:
            print(f"Cache metrics log failed: {e}", flush=True)

    def get_embeddings_for_models(self, paper_ids: List[str], model_names: List[str]) -> Dict[str, Dict[str, List[float]]]:
        """
        Retrieve embeddings for the given paper_ids across multiple models in a
        single dataset scan.

        Returns a dict mapping model_name -> {paper_id -> embedding}.
        """
        result = {model_name: {} for model_name in model_names}
        if not paper_ids or not model_names:
            return result

        unique_pids = list(set(paper_ids))

        try:
            dataset = self._get_dataset()
            table = dataset.to_table(
                columns=["paper_id", "embedding", "model"],
                filter=(
                    pc.field("model").isin(model_names)
                    & pc.field("paper_id").isin(unique_pids)
                )
            )
        except Exception:
            return result

        rows = table.to_pydict()
        for pid, emb, model in zip(rows["paper_id"], rows["embedding"], rows["model"]):
            if model in result:
                result[model][pid] = emb
        return result

    def get_embeddings(self, paper_ids: List[str], model_name: str) -> Dict[str, List[float]]:
        """
        Retrieve embeddings for the given paper_ids from the global cache.

        Returns a dict mapping paper_id -> embedding (list of floats).
        """
        result = self.get_embeddings_for_models(paper_ids, [model_name])
        return result.get(model_name, {})

    def write_cache_file(self, paper_ids: List[str], model_name: str, dest_path: str,
                         job_id: str = None) -> int:
        """
        Query the global cache and write the results to a local JSONL file
        in the format expected by Predictor._load_cached_publication_embeddings.
        Returns the number of embeddings written.
        """
        total_start = time.time()
        unique_pids = list(set(paper_ids))

        scan_start = time.time()
        embeddings = self.get_embeddings(paper_ids, model_name)
        scan_time_s = time.time() - scan_start

        serialize_start = time.time()
        if not embeddings:
            return 0
        rows = list(embeddings.items())
        serialize_time_s = time.time() - serialize_start

        write_start = time.time()
        with open(dest_path, 'w') as f:
            for pid, emb in rows:
                f.write(json.dumps({"paper_id": pid, "embedding": emb}) + '\n')
        write_time_s = time.time() - write_start
        total_time_s = time.time() - total_start

        if job_id:
            self._log_metrics(
                job_id=job_id,
                model_name=model_name,
                paper_count=len(unique_pids),
                cached_count=len(embeddings),
                scan_time_s=scan_time_s,
                serialize_time_s=serialize_time_s,
                write_time_s=write_time_s,
                total_time_s=total_time_s,
            )

        return len(embeddings)
