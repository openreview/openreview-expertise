from expertise.preprocessors.textrank import TextRank

def keyphrases(text, include_scores=False):
	textrank = TextRank()
	textrank.analyze(text, chunks=False)
	return [(word, _) if include_scores else word for word, _ in textrank.keyphrases()]
