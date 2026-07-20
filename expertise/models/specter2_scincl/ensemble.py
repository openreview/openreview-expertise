import os
import torch

from .specter import Specter2Predictor
from .scincl import SciNCLPredictor
from tqdm import tqdm


class EnsembleModel:
    def __init__(self, specter_dir, work_dir,
                 average_score=False, max_score=True, specter_batch_size=16, merge_alpha=0.5,
                 use_cuda=True, sparse_value=None, use_redis=False, compute_paper_paper=False, percentile_select=None, venue_specific_weights=None, normalize_scores=True,
                 specter2_hf_dir=None, specter2_adapter_dir=None, scincl_hf_dir=None):
        self.specter_predictor = Specter2Predictor(
            specter_dir=specter_dir,
            work_dir=os.path.join(work_dir, "specter"),
            average_score=average_score,
            max_score=max_score,
            batch_size=specter_batch_size,
            use_cuda=use_cuda,
            sparse_value=sparse_value,
            use_redis=use_redis,
            compute_paper_paper=compute_paper_paper,
            venue_specific_weights=venue_specific_weights,
            percentile_select=percentile_select,
            normalize_scores=normalize_scores,
            specter2_hf_dir=specter2_hf_dir,
            specter2_adapter_dir=specter2_adapter_dir
        )

        self.scincl_predictor = SciNCLPredictor(
            specter_dir=specter_dir,
            work_dir=os.path.join(work_dir, "scincl"),
            average_score=average_score,
            max_score=max_score,
            batch_size=specter_batch_size,
            use_cuda=use_cuda,
            sparse_value=sparse_value,
            use_redis=use_redis,
            compute_paper_paper=compute_paper_paper,
            venue_specific_weights=venue_specific_weights,
            percentile_select=percentile_select,
            normalize_scores=normalize_scores,
            scincl_hf_dir=scincl_hf_dir
        )
        self.merge_alpha = merge_alpha
        self.sparse_value = sparse_value
        self.normalize_scores = normalize_scores
        self.scores_matrix = None
        self.test_id_list = None
        self.reviewer_ids = None

    def set_archives_dataset(self, archives_dataset):
        print("Setting SPECTER archives")
        self.specter_predictor.set_archives_dataset(archives_dataset)
        print("Setting SciNCL archives")
        self.scincl_predictor.set_archives_dataset(archives_dataset)

    def set_submissions_dataset(self, submissions_dataset):
        print("Setting SPECTER submissions")
        self.specter_predictor.set_submissions_dataset(submissions_dataset)
        print("Setting SciNCL submissions")
        self.scincl_predictor.set_submissions_dataset(submissions_dataset)

    def all_scores(self, specter_publications_path=None, scincl_publications_path=None,
                   specter_submissions_path=None, scincl_submissions_path=None,
                   matrix_path=None):
        print("SPECTER:", flush=True)
        # Components compute their score matrix in memory; we don't persist
        # per-component matrices to disk (matrix_path=None) because the
        # ensemble only needs the merged result for downstream consumers.
        self.specter_predictor.all_scores(
            publications_path=specter_publications_path,
            submissions_path=specter_submissions_path,
            matrix_path=None,
        )
        print("SciNCL:", flush=True)
        self.scincl_predictor.all_scores(
            publications_path=scincl_publications_path,
            submissions_path=scincl_submissions_path,
            matrix_path=None,
        )

        # Matrix-level merge replaces the per-tuple merge that previously
        # iterated ~175M (test, reviewer) pairs in Python. The two component
        # matrices share row/col ordering because both predictors were fed
        # the same archives/submissions datasets.
        print("EnsembleModel: matrix-level merge...", flush=True)
        specter_m = self.specter_predictor.scores_matrix
        scincl_m = self.scincl_predictor.scores_matrix
        merged = specter_m * self.merge_alpha + scincl_m * (1 - self.merge_alpha)
        if self.normalize_scores:
            merged = torch.clamp(merged, 0.0, 1.0)
        else:
            merged = torch.clamp(merged, -1.0, 1.0)
        # Round once, vectorized — matches the previous per-row round(..., 4).
        merged = (merged * 10000).round() / 10000

        self.scores_matrix = merged
        self.test_id_list = self.specter_predictor.test_id_list
        self.reviewer_ids = self.specter_predictor.reviewer_ids

        if matrix_path:
            print(f"Saving ensemble scores matrix to {matrix_path}...", flush=True)
            torch.save({
                'scores': self.scores_matrix,
                'test_ids': self.test_id_list,
                'reviewer_ids': self.reviewer_ids,
            }, matrix_path)

        return self.scores_matrix
