"""
Huggingface_stats.py

Helper functions to preprocess/analyse text articles
"""

import nltk
import itertools
import re
from collections import Counter
from tqdm import tqdm

class Preprocessor:
    """
    Used to preprocess raw text such as news articles

    stemmer: NLTK stemmer, e.g. EnglishStemmer()
    stopwords: list of uninformative words, e.g ['a', 'the', 'an']
    re_pattern: compiled regular expression pattern that
    words must match, e.g. re.compile("[a-zA-Z]")
    """
    def __init__(self, stemmer=None, stopwords=None, re_pattern=None):
        self._stemmer = stemmer
        self._re_pattern = re_pattern

        if stemmer is not None and stopwords is not None:
            assert isinstance(stopwords, list), 'stopwords must be a list'
            # stem all stopwords
            self._stopwords = set([stemmer.stem(w) for w in stopwords])
        else:
            self._stopwords = set(stopwords)

    def word_split(self, doc: str):
        """
        takes a document as input, returns a list of all words in that
        document

        doc: string
        """
        sents = nltk.tokenize.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sents]
        words = list(itertools.chain(*words))

        # stem the words. If no stemmer was given, set all
        # words to lowercase
        if self._stemmer is not None:
            words = [self._stemmer.stem(w) for w in words]
        else:
            words = [w.lower() for w in words]

        # filter out stopwords
        if self._stopwords is not None:
            words = [w for w in words if w not in self._stopwords]

        # filter out words that do not match a regular expression
        if self._re_pattern is not None:
            words = [w for w in words if self._re_pattern.match(w)]

        return words


    def word_count(self, doc: str):
        """
        returns a Counter with all words in a document

        doc: string
        """
        return Counter(self.word_split(doc))


    def batch_word_count(self, docs: list, show_progress=False):
        """
        returns a Counter with all words in a list of documents

        docs: list of strings
        show_progress: boolean, controls whether or not a progress bar shows
        """
        words = Counter()

        # count all words with a visual progress bar
        if show_progress:
            for i in tqdm(range(len(docs))):
                words += self.word_count(docs[i])
        # count all words without a visual progress bar
        else:
            for doc in docs:
                words += self.word_count(doc)
        return words

    # helper function for document_frequency
    def _document_frequency_helper(self, doc: str):
        words = list(set(self.word_split(doc)))
        return Counter(words)


    def document_frequency(self, docs: list, show_progress=False):
        """
        calculates a dictionary where each key is a word in the dataset
        and each value indicates how many documents contain that word

        docs: list of strings
        show_progress: boolean, controls whether or not a progress bar shows
        """
        freqs = Counter()

        # count all words with a visual progress bar
        if show_progress:
            for i in tqdm(range(len(docs))):
                freqs += self._document_frequency_helper(docs[i])
        # count all words without a visual progress bar
        else:
            for doc in docs:
                freqs += self._document_frequency_helper(doc)
        return freqs
