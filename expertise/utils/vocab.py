import codecs
import numpy as np
from collections import Counter
import csv
import io

from expertise import utils

# Note, because I used sets when I constructed the reviewer keyphrase file, I end up with
# count_keyphrases being a count based on number of documents containing the keyphrase.
def count_keyphrases(reviewer_keyphrase_file, submission_keyphrase_file, outputfile, min_count):

    keyphrase_counter = Counter()

    with open(reviewer_keyphrase_file, 'rb') as f:
        reviewer_keyphrases = pickle.load(f)

    for kp_list in reviewer_keyphrases.values():
        keyphrase_counter.update(kp_list)

    with open(submission_keyphrase_file, 'rb') as f:
        submission_keyphrases = pickle.load(f)

    for kp_list in submission_keyphrases.values():
        keyphrase_counter.update(kp_list)

    # start at 2 because other indices are reserved (padding and OOV)
    kp_index = 2
    with open(outputfile, 'w') as f:
        writer = csv.writer(delimiter='\t')
        for keyphrase, count in keyphrase_counter.items():
            if count >= min_count:
                writer.writerow([keyphrase, kp_index])
                kp_index += 1

class Vocab(object):
    def __init__(self, vocabfile=None, min_count=1):
        self.index_by_item = {}
        self.item_by_index = {}
        self.count_by_item = Counter()

        self.OOV = "<OOV>"

        self.PADDING_INDEX = 0
        self.OOV_INDEX = 1
        self.next_index = 2

        self.index_by_item[self.OOV] = self.OOV_INDEX
        self.item_by_index[self.OOV_INDEX] = self.OOV

        self.min_count = int(min_count)

        if vocabfile:
            self.vocabfile = vocabfile
            with open(self.vocabfile) as f:
                reader = csv.reader(f, delimiter='\t')
                for idx_str, item in reader:
                    index = int(idx_str)
                    self.index_by_item[item] = index
                    self.item_by_index[index] = item
                    self.next_index = index + 1
                    self.count_by_item.update([item])

    def __len__(self):
        return len(self.index_by_item)

    def dump_csv(self, outfile=None, delimiter='\t', encoding='utf-8'):
        '''
        writes the vocab to a csv file
        '''

        def write_to_buffer(output):
            writer = csv.writer(output, delimiter=delimiter)
            for item, count in self.count_by_item.items():
                if count >= self.min_count:
                    writer.writerow([self.index_by_item[item], item])

        if outfile:
            with open(outfile, 'w') as f:
                write_to_buffer(f)

        output = io.StringIO()
        write_to_buffer(output)

        csv_binary = output.getvalue().encode(encoding)

        return csv_binary

    def load_items(self, vocab_items):
        assert self.next_index not in self.item_by_index, \
            'self.next_index not properly incremented (this should not happen)'

        for item in vocab_items:
            if item not in self.index_by_item:
                self.index_by_item[item] = self.next_index
                self.item_by_index[self.next_index] = item
                self.next_index += 1

        self.count_by_item.update(vocab_items)

    def to_ints(self, kp_list, length=None):
        kp_indices = [self.index_by_item.get(kp, self.OOV_INDEX) for kp in kp_list]

        if length:
            return utils.fixedwidth(kp_indices, length, pad_val=self.PADDING_INDEX)
        else:
            return kp_indices

    def to_string(self,list_ints):
        stri = ""
        for c in list_ints:
            if c != self.PADDING_INDEX:
                stri += self.item_by_index.get(c,self.OOV).encode("utf-8") + " "
        return stri
