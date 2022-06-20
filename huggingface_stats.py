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
        self._positive_keywords = []
        self._negative_keywords = []
        self._positive_keywords_threshold = 0
        self._negative_keywords_threshold = 0

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


    def stem_word_list(self, words: list):
        """
        Stems a list of words. If stemmer is None, words will be made
        lower case.
        """
        if self._stemmer is not None:
            words = [self._stemmer.stem(w) for w in words]
        else:
            words = [w.lower() for w in words]
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

    def init_keyword_filter(self, positive: list, pos_threshold: int, negative: list, neg_threshold: int):
        """
        Initialises the Preprocessor's keyword filter. Must happen
        after the __init__ since the keywords are preprocessed by the
        same Preprocessor.

        positive: list of keywords that should appear in the document
        pos_threshold: minimum amount of positive keywords that the document must contain
        negative: list of keywords that should not appear in the document
        neg_threshold: maximum amount of negative keywords that the document may contain

        """
        # convert to set in case multiple keywords get stemmed to the same stem
        self._positive_keywords = list(set(self.stem_word_list(positive)))
        self._negative_keywords = list(set(self.stem_word_list(negative)))
        self._positive_keywords_threshold = pos_threshold
        self._negative_keywords_threshold = neg_threshold

    def keyword_filter(self, doc: str):
        """
        Returns a boolean value based on whether or not a document contains
        certain keywords.

        doc: string, document to check
        """
        words = set(self.word_split(doc))
        # count how many positive keywords occur at least once in the document
        pos_in_doc = 0
        for word in self._positive_keywords:
            if word in words:
                pos_in_doc += 1
        # count how many positive keywords occur at least once in the document
        neg_in_doc = 0
        for word in self._negative_keywords:
            if word in words:
                neg_in_doc += 1

        if pos_in_doc >= self._positive_keywords_threshold and neg_in_doc < self._negative_keywords_threshold:
            return True
        return False

    def batch_keyword_filter(self, docs: list, show_progress=False):
        """
        Applies the keyword_filter to a list of documents. Returns a
        list of boolean values the same length as docs.

        docs: list of documents to check
        show_progress: boolean, controls whether or not a progress bar shows
        """
        matches = list()

        # filter all docs with a visual progress bar
        if show_progress:
            for i in tqdm(range(len(docs))):
                matches.append(self.keyword_filter(docs[i]))
        # filter all docs without a visual progress bar
        else:
            matches = [self.keyword_filter(doc) for doc in docs]
        return matches

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
        return score#/len(doc) # normalise for document length

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
