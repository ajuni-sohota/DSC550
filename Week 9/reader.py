#!/usr/bin/env python3

import nltk
import pickle
import sqlite3

from nltk.corpus.reader.api import CorpusReader

PKL_PATTERN = r'(?!\.)[\w\s\d\-]+\.pickle'

class SqliteCorpusReader(object):

    def __init__(self, path):
        self._cur = sqlite3.connect(path).cursor()

    def text(self):
        self._cur.execute("SELECT txt FROM categorized_comments")
        texts = self._cur.fetchall()
        for text in texts:
            yield text

    def categories(self):
        self._cur.execute("SELECT cat FROM categorized_comments")
        cats = self._cur.fetchall()
        for cat in cats:
            yield cat
    def paras(self):
        """
        Returns a generator of paragraphs.
        """
        for text in self.texts():
            for paragraph in text:
                yield paragraph

    def sents(self):
        """
        Returns a generator of sentences.
        """
        for para in self.paras():
            for sentence in nltk.sent_tokenize(para):
                yield sentence

    def words(self):
        """
        Returns a generator of words.
        """
        for sent in self.sents():
            for word in nltk.wordpunct_tokenize(sent):
                yield word

    def tagged_tokens(self):
        for sent in self.sents():
            for word in nltk.wordpunct_tokenize(sent):
                yield nltk.pos_tag(word)



class PickledReviewsReader(CorpusReader):
    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        """
        Initialize the corpus reader
        """
        CorpusReader.__init__(self, root, fileids, **kwargs)

    def texts_scores(self, fileids=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the SqliteCorpusReader, this uses a generator
        to achieve memory safe iteration.
        """
        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def reviews(self, fileids=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of (token, tag) tuples.
        """
        for text,score in self.texts_scores(fileids):
            yield text

    def scores(self, fileids=None):
        """
        Return the scores
        """
        for text,score in self.texts_scores(fileids):
            yield score

    def paras(self, fileids=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of (token, tag) tuples.
        """
        for review in self.reviews(fileids):
            for paragraph in review:
                yield paragraph

    def sents(self, fileids=None):
        """
        Returns a generator of sentences where each sentence is a list of
        (token, tag) tuples.
        """
        for paragraph in self.paras(fileids):
            for sentence in paragraph:
                yield sentence

    def tagged(self, fileids=None):
        for sent in self.sents(fileids):
            for token in sent:
                yield token

    def words(self, fileids=None):
        """
        Returns a generator of (token, tag) tuples.
        """
        for token in self.tagged(fileids):
            yield token[0]


# if __name__ == '__main__':
#     # Download the data from https://www.kaggle.com/nolanbconaway/pitchfork-data/data
#     # preprocess by running preprocess.py to produce pickled corpus
#     reader = PickledReviewsReader('../review_corpus_proc')
#     print(len(list(reader.reviews())))
