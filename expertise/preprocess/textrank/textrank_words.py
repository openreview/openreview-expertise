from .textrank import TextRank


def keyphrases(text, include_scores=False, include_tokenlist=False):
    textrank = TextRank()
    textrank.analyze(text, chunks=False)
    tokenlist = [word for sentence in textrank.sentences for word in sentence]
    top_tokens = [
        (word, _) if include_scores else word for word, _ in textrank.keyphrases()
    ]

    if include_tokenlist:
        return top_tokens, tokenlist
    else:
        return top_tokens
