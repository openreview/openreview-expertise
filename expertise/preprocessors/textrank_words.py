from expertise.preprocessors.textrank import TextRank

def keyphrases(text):
	textrank = TextRank()
	textrank.analyze(text, chunks=False)
	return [word for word, _ in textrank.keyphrases()]
