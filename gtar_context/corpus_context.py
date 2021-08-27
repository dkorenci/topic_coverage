'''
Creation of corpora for "Getting the Agenda Right" (gtar) corpora.
'''

from pyutils.file_utils.location import FolderLocation

from pytopia.context.Context import Context
from pytopia.corpus.postgres.feedsucker import *
from pytopia.corpus.filter.duplicate import DuplicateTextFilter
from pytopia.corpus.filter.FilteredCorpus import FilteredCorpus
from pytopia.corpus.filter.textsize import TextsizeFilter
from pytopia.nlp.text2tokens.regexp import alphanumTokenizer

import pickle
from os import path

thisfolder = path.dirname(__file__)
resfolder = FolderLocation(path.abspath(path.join(thisfolder, 'corpus_resources')))

def test():
    from pytopia.testing.validity_checks import checkCorpus
    ctx = gtarCorpusContext()
    for corp in ctx:
        checkCorpus(corp)

def gtarTextPerLineCorpus():
    from pytopia.corpus.text.TextPerLineCorpus import TextPerLineCorpus
    from gtar_context.settings import gtar_text_per_line_corpus
    fname = FolderLocation(thisfolder)(gtar_text_per_line_corpus)
    corpus = TextPerLineCorpus(fname, id='us_politics_textperline')
    return corpus

def gtarCorpusContext():
    ctx = Context('gtar_corpus_context')
    builder = GtarCorpusBuilder('rsssucker_topus1_13042015',
                                resfolder(), '2015-01-26 00:00:00')
    # orig. fulltext corpora
    # to work, a database with orig. GtAR data is needed
    # for name in dir(builder):
    #     if name.startswith('corpus', 0):
    #         method = getattr(builder, name)
    #         corpus = method()
    #         ctx.add(corpus)
    ctx.add(gtarTextPerLineCorpus())
    ctx.add(testCorpus())
    return ctx

from pytopia.corpus.text.TextPerLineCorpus import TextPerLineCorpus
def testCorpus():
    '''Small corpus for quick testing.'''
    # todo solve problem with test corpus
    f = resfolder('us_politics_dedup_1000.txt')
    c = TextPerLineCorpus(f)
    c.id = 'gtar_test_corpus'
    return c

class GtarCorpusBuilder():

    def __init__(self, database, resfolder, startdate):
        '''
        :param database: feedsucker database on which to base corpuses
        :param resfolder: folder with feed lists and id sets
        '''
        self.dbase = database
        self.resourceFolder = resfolder
        self.startdate = startdate

    def __prefilter(self):
        '''Create and return text filter of pre-filtering all texts,
        that filters out texts with less then 40 alphanumeric tokens
        '''
        return TextsizeFilter(40, alphanumTokenizer())

    def __baseCorpus(self):
        '''Create and return corpus with all the texts in the database (no filtering). '''
        return FeedsuckerCorpus(self.dbase)

    def __dupFilter(self):
        '''Create and return duplicate text filter based on base corpus. '''
        return DuplicateTextFilter(self.__baseCorpus())

    def  __feedlist(self, feedFile):
        '''Load list of feeds from file located in feedfolder. '''
        return Feedlist(path.join(self.resourceFolder, feedFile))

    def __feedsetCorpus(self, feedFile):
        '''Create and return feedset corpus based on a feedlist and database.
         :param feedFile: name of the file (inside feedfolder) with a list of feeds
        '''
        return FeedsetCorpus(self.dbase, self.__feedlist(feedFile),
                             startDate='2015-01-26 00:00:00')

    def corpusUsNews(self):
        '''Corpus of texts from feeds related to US'''
        return FilteredCorpus('us_news', self.__feedsetCorpus('us_news_feeds.txt'),
                              [self.__prefilter(), self.__dupFilter()])

    def corpusUsPolitics(self):
        '''us_politics, corpus for all GtAR experiments, contains texts
         from feeds about US politics. '''
        return FilteredCorpus('us_politics', self.__feedsetCorpus('us_politics_feeds.txt'),
                              [self.__prefilter(), self.__dupFilter()])

    def corpusUsPolitics_raw(self):
        '''All us_politics feeds texts, no filtering or deduplication'''
        return FeedsetCorpus(id='us_politics_raw', dbName=self.dbase,
                             feedlist=self.__feedlist('us_politics_feeds.txt'),
                             startDate='2015-01-26 00:00:00')

    def corpusUsPolitics_nodedup(self):
        '''us_politics, no deduplication, just prefiltering'''
        return FilteredCorpus('us_politics_nodedup', self.__feedsetCorpus('us_politics_feeds.txt'),
                              [self.__prefilter()])

    def corpusUsPolitics_dedup(self):
        '''
        Fixed id set of ids created from us_politics_nodedup
         by applying levenshtein clustering and agenda dedup.
        '''
        f = path.join(self.resourceFolder,
                      'dedupIds_us_politics_nodedup_dedup_lev_clusterer_agenda_dedup.pickle')
        idSet = pickle.load(open(f, 'rb'))
        return IdsetCorpus('us_politics_dedup', self.dbase, idSet)

if __name__ == '__main__':
    test()