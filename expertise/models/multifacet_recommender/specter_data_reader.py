from tokenize import Token
from typing import Optional, Tuple, List

from allennlp.data import DatasetReader
from overrides import overrides
from specter.data import DataReader, NO_VENUE_TEXT
from specter.data_utils.create_training_files import get_text_tokens


@DatasetReader.register("specter_data_reader_custom")
class SpecterDataReader(DataReader):

    @overrides
    def _get_paper_features(self, paper: Optional[dict] = None) -> Tuple[List[Token], List[Token], List[Token], int, List[Token]]:
        """ Given a paper, extract and tokenize abstract, title, venue and year"""
        if paper:
            paper_id = paper.get('paper_id')

            # This function is being called by the same paper multiple times.
            # Cache the result to avoid wasted compute
            if self.use_paper_feature_cache and paper_id in self.paper_feature_cache:
                return self.paper_feature_cache[paper_id]

            if not self.concat_title_abstract:
                abstract_tokens = self._tokenizer.tokenize(paper.get('abstract') or '.')
                title_tokens = self._tokenizer.tokenize(paper.get('title') or '.')
                if self.max_sequence_length > 0:
                    title_tokens = title_tokens[:self.max_sequence_length]
                    abstract_tokens = abstract_tokens[:self.max_sequence_length]
            else:
                abstract_tokens = self._tokenizer.tokenize(paper.get("abstract") or ".")
                title_tokens = self._tokenizer.tokenize(paper.get("title") or ".")
                if 'abstract' in self.included_text_fields:
                    title_tokens = get_text_tokens(title_tokens, abstract_tokens, self.abstract_delimiter)
                if 'authors' in self.included_text_fields:
                    author_text = ' '.join(paper.get("author-names") or [])
                    author_tokens = self._tokenizer.tokenize(author_text)
                    max_seq_len_title = self.max_sequence_length - 15  # reserve max 15 tokens for author names
                    title_tokens = title_tokens[:max_seq_len_title] + self.author_delimiter + author_tokens
                title_tokens = title_tokens[:self.max_sequence_length]
                # abstract and title are identical (abstract won't be used in this case)
                abstract_tokens = title_tokens

            venue = self._tokenizer.tokenize(paper.get('venue') or NO_VENUE_TEXT)
            year = paper.get('year') or 0
            body_tokens = self._tokenizer.tokenize(paper.get('body')) if 'body' in paper else None
            features = abstract_tokens, title_tokens, venue, year, body_tokens

            if self.use_paper_feature_cache:
                self.paper_feature_cache[paper_id] = features

            return features
        else:
            return None, None, None, None, None