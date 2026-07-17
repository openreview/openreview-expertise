from typing import List, Dict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


class GlobalEmbeddingsCache:
    """
    Reads embeddings from a Hive-partitioned Parquet store on GCS.

    Expected layout:
        gs://<bucket>/<prefix>/model=<model>/year_month=<YYYY-MM>/job_id=<job_id>/*.parquet

    Schema columns: paper_id (string), embedding (list<float>), embedding_date (timestamp[ms]), model, year_month, job_id
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

    def get_embeddings_for_models(self, paper_ids: List[str], model_names: List[str],
                                    paper_mdates: Dict[str, str] = None) -> Dict[str, Dict[str, List[float]]]:
        """
        Retrieve embeddings for the given paper_ids across multiple models in a
        single dataset scan. If paper_mdates is provided, only return embeddings
        whose cached embedding_date is greater than or equal to the paper's mdate.

        Returns a dict mapping model_name -> {paper_id -> embedding}.
        """
        result = {model_name: {} for model_name in model_names}
        if not paper_ids or not model_names:
            return result

        unique_pids = list(set(paper_ids))

        try:
            dataset = self._get_dataset()
            table = dataset.to_table(
                columns=["paper_id", "embedding", "model", "embedding_date"],
                filter=(
                    pc.field("model").isin(model_names)
                    & pc.field("paper_id").isin(unique_pids)
                )
            )
        except Exception:
            return result

        if paper_mdates and len(table) > 0:
            mdate_list = [paper_mdates.get(pid.as_py(), None) for pid in table["paper_id"]]
            requested = pa.array(mdate_list, pa.int64())
            ts_col = pc.cast(requested, pa.timestamp('ms'))
            table = table.append_column("requested_mdate", ts_col)
            keep = pc.or_(
                pc.is_null(table["requested_mdate"]),
                pc.or_(
                    pc.is_null(table["embedding_date"]),
                    pc.greater_equal(table["embedding_date"], table["requested_mdate"])
                )
            )
            keep = pc.if_else(pc.is_null(keep), True, keep)
            table = table.filter(keep)

        for model in model_names:
            model_table = table.filter(pc.equal(table["model"], model))
            if len(model_table) > 0:
                rows = model_table.to_pydict()
                result[model] = dict(zip(rows["paper_id"], rows["embedding"]))
        return result

    def get_embeddings(self, paper_ids: List[str], model_name: str,
                       paper_mdates: Dict[str, str] = None) -> Dict[str, List[float]]:
        """
        Retrieve embeddings for the given paper_ids from the global cache.

        Returns a dict mapping paper_id -> embedding (list of floats).
        """
        result = self.get_embeddings_for_models(paper_ids, [model_name], paper_mdates=paper_mdates)
        return result.get(model_name, {})
