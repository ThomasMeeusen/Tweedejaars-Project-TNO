"""
Huggingface_stats.py

Helper functions to preprocess/analyse text articles
"""

import nltk
import itertools
import re
from collections import Counter
from tqdm import tqdm
import numpy as np

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

class tf_idf_document_scorer:
    """
    Calculates a score for a document that represents how likely
    the document fits the positive class, based on the average
    tf-idf per class for each word

    preprocessor: preprocessor object
    docs: list of documents to train the scorer
    labels: list of labels (1 or 0) that belong to the documents
    show_progress: boolean, controls whether or not a progress bar shows
    """
    def __init__(self, preprocessor: Preprocessor, docs: list, labels: list, show_progress=False):
        self._preprocessor = preprocessor

        # learn the word weights
        doc_freq = preprocessor.document_frequency(docs=docs, show_progress=show_progress)
        vocab = list(dict(doc_freq.items()).keys())

        docs_positive = np.array(docs)[np.array(labels) == 1].tolist()
        docs_negative = np.array(docs)[np.array(labels) == 0].tolist()
        summed_tf_positive = preprocessor.batch_word_count(docs=docs_positive, show_progress=show_progress)
        summed_tf_negative = preprocessor.batch_word_count(docs=docs_negative, show_progress=show_progress)

        N = len(docs)
        n_positive = len(docs_positive)
        n_negative = len(docs_negative)

        self._weights = dict()
        for word in vocab:
            avg_tf_pos = summed_tf_positive[word] / n_positive
            avg_tf_neg = summed_tf_negative[word] / n_negative
            idf = np.log(N / (1 + doc_freq[word]))
            self._weights[word] = (avg_tf_pos - avg_tf_neg) * idf

    def score(self, doc: str):
        """
        Computes a score for a document, based on a bag-of-words approach.
        The higher the score, the more it fits the positive documents in
        the training data.
        """
        word_count = self._preprocessor.word_count(doc=doc)
        score = 0
        for word, count in word_count.items():
            if word in self._weights:
                score += count * self._weights[word]
        return score

    def batch_score(self, docs: list, show_progress=False):
        """
        returns a list of scores for the documents

        docs: list of strings, documents to score
        show_progress: boolean, controls whether or not a progress bar shows
        """
        scores = []
        # score all docs with a visual progress bar
        if show_progress:
            for i in tqdm(range(len(docs))):
                scores.append(self.score(docs[i]))
        # score all docs without a visual progress bar
        else:
            for doc in docs:
                scores.append(self.score(doc))

        return scores
