from expertise.preprocessors.textrank import TextRank

def keyphrases(text):
    textrank = TextRank()
    chunked_sentences = textrank.sentence_segment_chunk(text)
    return [word for sentence in chunked_sentences for word in sentence]
