from expertise.preprocess.textrank import TextRank


def keyphrases(text):
    textrank = TextRank()
    segmented_sentences = textrank.sentence_segment(text)
    return [word for sentence in segmented_sentences for word in sentence]
