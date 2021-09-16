from .utils import unk_string, lookup


class Instance(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.embeddings = []

    def populate_ngrams(self, words, zero_unk, n):
        embeddings = []
        if type(self.sentence) == str:
            sentence = [self.sentence]
        else:
            sentence = self.sentence
        for i in sentence:
            sent = " " + i.strip() + " "

            for j in range(len(sent)):
                idx = j
                gr = ""
                while idx < j + n and idx < len(sent):
                    gr += sent[idx]
                    idx += 1
                if not len(gr) == n:
                    continue
                wd = lookup(words, gr, zero_unk)
                if wd is not None:
                    embeddings.append(wd)

        if len(embeddings) == 0:
            return [words[unk_string]]
        return embeddings

    def populate_embeddings(self, words, zero_unk, ngrams):
        if ngrams:
            self.embeddings = self.populate_ngrams(words, zero_unk, ngrams)
        else:
            if type(self.sentence) == str:
                sentence = [self.sentence]
            else:
                sentence = self.sentence
            for i in sentence:
                arr = i.split()
                for i in arr:
                    wd = lookup(words, i, zero_unk)
                    if wd is not None:
                        self.embeddings.append(wd)
            if len(self.embeddings) == 0:
                self.embeddings = [words[unk_string]]
