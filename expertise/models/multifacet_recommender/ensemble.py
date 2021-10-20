import os

from .specter import SpecterPredictor
from .multifacet_recommender import MultiFacetRecommender
from tqdm import tqdm


class EnsembleModel:
    def __init__(
        self,
        specter_dir,
        mfr_feature_vocab_file,
        mfr_checkpoint_dir,
        mfr_epochs,
        work_dir,
        average_score=False,
        max_score=True,
        specter_batch_size=16,
        mfr_batch_size=50,
        merge_alpha=0.8,
        use_cuda=True,
        sparse_value=None,
    ):
        self.specter_predictor = SpecterPredictor(
            specter_dir=specter_dir,
            work_dir=os.path.join(work_dir, "specter"),
            average_score=average_score,
            max_score=max_score,
            batch_size=specter_batch_size,
            use_cuda=use_cuda,
            sparse_value=sparse_value,
        )

        self.mfr_predictor = MultiFacetRecommender(
            work_dir=os.path.join(work_dir, "mfr"),
            feature_vocab_file=mfr_feature_vocab_file,
            model_checkpoint_dir=mfr_checkpoint_dir,
            epochs=mfr_epochs,
            batch_size=mfr_batch_size,
            use_cuda=use_cuda,
            sparse_value=sparse_value,
        )

        self.merge_alpha = merge_alpha
        self.sparse_value = sparse_value
        self.preliminary_scores = None

    def set_archives_dataset(self, archives_dataset):
        print("Setting SPECTER archives")
        self.specter_predictor.set_archives_dataset(archives_dataset)
        print("Setting MultiFacetRecommender archives")
        self.mfr_predictor.set_archives_dataset(archives_dataset)

    def set_submissions_dataset(self, submissions_dataset):
        print("Setting SPECTER submissions")
        self.specter_predictor.set_submissions_dataset(submissions_dataset)
        print("Setting MultiFacetRecommender submissions")
        self.mfr_predictor.set_submissions_dataset(submissions_dataset)

    def embed_submissions(
        self,
        specter_submissions_path=None,
        mfr_submissions_path=None,
        skip_specter=False,
    ):
        if not skip_specter:
            print("SPECTER:")
            self.specter_predictor.embed_submissions(specter_submissions_path)
        print("MFR:")
        self.mfr_predictor.embed_submissions(mfr_submissions_path)

    def embed_publications(
        self,
        specter_publications_path=None,
        mfr_publications_path=None,
        skip_specter=False,
    ):
        if not skip_specter:
            print("SPECTER:")
            self.specter_predictor.embed_publications(specter_publications_path)
        print("MFR:")
        self.mfr_predictor.embed_publications(mfr_publications_path)

    def all_scores(
        self,
        specter_publications_path=None,
        mfr_publications_path=None,
        specter_submissions_path=None,
        mfr_submissions_path=None,
        scores_path=None,
    ):
        print("SPECTER:")
        specter_scores_path = os.path.join(
            self.specter_predictor.work_dir, "specter_affinity.csv"
        )
        self.specter_predictor.all_scores(
            specter_publications_path, specter_submissions_path, specter_scores_path
        )
        print("MFR:")
        mfr_scores_path = os.path.join(self.mfr_predictor.work_dir, "mfr_affinity.csv")
        self.mfr_predictor.all_scores(
            mfr_publications_path, mfr_submissions_path, mfr_scores_path
        )

        # Convert preliminary scores of SPECTER to a dictionary
        csv_scores = []
        self.preliminary_scores = []
        specter_preliminary_scores_map = {}
        for entry in self.specter_predictor.preliminary_scores:
            specter_preliminary_scores_map[(entry[0], entry[1])] = entry[2]

        for entry in self.mfr_predictor.preliminary_scores:
            new_score = specter_preliminary_scores_map[
                (entry[0], entry[1])
            ] * self.merge_alpha + entry[2] * (1 - self.merge_alpha)
            csv_line = "{note_id},{reviewer},{score}".format(
                note_id=entry[0], reviewer=entry[1], score=new_score
            )
            csv_scores.append(csv_line)
            self.preliminary_scores.append((entry[0], entry[1], new_score))

        if scores_path:
            with open(scores_path, "w") as f:
                for csv_line in csv_scores:
                    f.write(csv_line + "\n")

        return self.preliminary_scores

    def _sparse_scores_helper(self, all_scores, id_index):
        counter = 0
        # Get the first note_id or profile_id
        current_id = self.preliminary_scores[0][id_index]
        if id_index == 0:
            desc = "Note IDs"
        else:
            desc = "Profiles IDs"
        for note_id, profile_id, score in tqdm(
            self.preliminary_scores, total=len(self.preliminary_scores), desc=desc
        ):
            if counter < self.sparse_value:
                all_scores.add((note_id, profile_id, score))
            elif (note_id, profile_id)[id_index] != current_id:
                counter = 0
                all_scores.add((note_id, profile_id, score))
                current_id = (note_id, profile_id)[id_index]
            counter += 1
        return all_scores

    def sparse_scores(self, scores_path=None):
        if self.preliminary_scores is None:
            raise RuntimeError("Call all_scores before calling sparse_scores")

        print("Sorting...")
        self.preliminary_scores.sort(key=lambda x: (x[0], x[2]), reverse=True)
        print("Sort 1 complete")
        all_scores = set()
        # They are first sorted by note_id
        all_scores = self._sparse_scores_helper(all_scores, 0)

        # Sort by profile_id
        print("Sorting...")
        self.preliminary_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        print("Sort 2 complete")
        all_scores = self._sparse_scores_helper(all_scores, 1)

        print("Final Sort...")
        all_scores = sorted(list(all_scores), key=lambda x: (x[0], x[2]), reverse=True)
        if scores_path:
            with open(scores_path, "w") as f:
                for note_id, profile_id, score in all_scores:
                    f.write("{0},{1},{2}\n".format(note_id, profile_id, score))

        print("Sparse score computation complete")
        return all_scores
