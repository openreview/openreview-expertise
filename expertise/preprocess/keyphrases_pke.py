import json
import os
import pickle
import pke
import string
import sys

from nltk.corpus import stopwords

from .. import utils


def keyphrases(data_dir):
    """
    Given a directory containing reviewer archives or submissions,
    generate a dict keyed on signatures whose values are sets of keyphrases.

    The input directory should contain .jsonl files. Files representing
    reviewer archives should be [...] TODO: Finish this.
    """

    reviewer_or_submission_keyphrases = {}

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        file_id = filename.replace(".jsonl", "")
        print(file_id)

        keyphrases = []

        with open(filepath) as f:
            for line in f.readlines():
                if line.endswith("\n"):
                    line = line[:-1]

                record = json.loads(line)
                content = record["content"]

                record_text_unfiltered = utils.content_to_text(
                    content, fields=["title", "abstract", "fulltext"]
                )
                record_text_filtered = utils.strip_nonalpha(record_text_unfiltered)

                # define the set of valid Part-of-Speeches
                pos = {"NOUN", "PROPN", "ADJ"}

                # 1. create a SingleRank extractor.
                extractor = pke.unsupervised.SingleRank()

                # 2. load the content of the document.
                extractor.load_document(
                    input=record_text_filtered, language="en", normalization=None
                )

                # 3. select the longest sequences of nouns and adjectives as candidates.
                extractor.candidate_selection(pos=pos)

                # 4. weight the candidates using the sum of their word's scores that are
                #    computed using random walk. In the graph, nodes are words of
                #    certain part-of-speech (nouns and adjectives) that are connected if
                #    they occur in a window of 10 words.
                extractor.candidate_weighting(window=10, pos=pos)

                # 5. get the 10-highest scored candidates as keyphrases
                keyphrases.extend(
                    [word[0].replace(" ", "_") for word in extractor.get_n_best(n=3)]
                )

        yield file_id, keyphrases
