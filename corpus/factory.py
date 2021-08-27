from filter import *
from rsssucker import *
from pymedialab_settings import settings

import os, pickle

class CorpusFactory():

    databases = settings.rsssucker_databases
    corpuses = None
    id2corpus = None

    base = os.path.dirname(__file__)
    res = os.path.abspath(os.path.join(base, '..', 'resources'))

    @staticmethod
    def getCorpusIds():
        if CorpusFactory.id2corpus is None: CorpusFactory.createCorpusList()
        return [c.corpusId() for c in CorpusFactory.corpuses]

    @staticmethod
    def getCorpus(id):
        if CorpusFactory.id2corpus is None: CorpusFactory.createCorpusList()
        return CorpusFactory.id2corpus[id]

    @staticmethod
    def createCorpusList():
        '''
        define list of corpuses to be used by the application
        '''
        CorpusFactory.corpuses = \
            [ CorpusFactory.usNewsCorpus(),
              CorpusFactory.usPoliticsCorpus(), CorpusFactory.usPoliticsCorpus_dedup(),
              CorpusFactory.usPoliticsCorpus_nodedup(), CorpusFactory.usPoliticsCorpus_raw(),
              CorpusFactory.usPoliticsCorpus_test(), CorpusFactory.usPoliticsCorpus_test_old(),
              CorpusFactory.usPoliticsCorpus_test_dedup(),
              CorpusFactory.worldNewsCorpus(),
              CorpusFactory.usNewsCorpus_test(),
              CorpusFactory.worldNewsCorpus_test() ]
        id2corpus = {}
        for c in CorpusFactory.corpuses:
            id2corpus[c.corpusId()] = c
        CorpusFactory.id2corpus = id2corpus

    @staticmethod
    def usNewsCorpus():
        f = os.path.join(CorpusFactory.res, 'us_news_feeds.txt')
        corpus = FeedsetCorpus('rsssucker_topus1_13042015', Feedlist(f))
        dupfilter = DuplicateTextFilter(RsssuckerCorpus('rsssucker_topus1_13042015'))
        return MultiFilteredCorpus(corpus, [RsssuckerFilter(), dupfilter], id='us_news')

    @staticmethod
    def usPoliticsCorpus():
        f = os.path.join(CorpusFactory.res, 'us_politics_feeds.txt')
        corpus = FeedsetCorpus('rsssucker_topus1_13042015', Feedlist(f))
        dupfilter = DuplicateTextFilter(RsssuckerCorpus('rsssucker_topus1_13042015'))
        return MultiFilteredCorpus(corpus, [RsssuckerFilter(), dupfilter], id='us_politics')

    @staticmethod
    def usPoliticsCorpus_raw():
        '''all texts from the database, no filtering or deduplication'''
        f = os.path.join(CorpusFactory.res, 'us_politics_feeds.txt')
        return FeedsetCorpus('rsssucker_topus1_13042015', Feedlist(f), id='us_politics_raw')

    @staticmethod
    def usPoliticsCorpus_dedup():
        '''
        Fixed id set of id created from us_politics_nodedup
         by applying levenshtein clustering and agenda dedup.
        '''
        f = os.path.join(CorpusFactory.res,
                         'dedupIds_us_politics_nodedup_dedup_lev_clusterer_agenda_dedup.pickle')
        idSet = pickle.load(open(f, 'rb'))
        dbase = 'rsssucker_topus1_13042015'
        return IdsetCorpus(dbase, idSet, id='us_politics_dedup')

    @staticmethod
    def usPoliticsCorpus_nodedup():
        f = os.path.join(CorpusFactory.res, 'us_politics_feeds.txt')
        corpus = FeedsetCorpus('rsssucker_topus1_13042015', Feedlist(f))
        return FilteredCorpus(corpus, RsssuckerFilter(), id='us_politics_nodedup')

    @staticmethod
    def usPoliticsCorpus_test():
        f = os.path.join(CorpusFactory.res, 'us_politics_test_feeds.txt')
        corpus = FeedsetCorpus('rsssucker_topus1_13042015', Feedlist(f))
        return FilteredCorpus(corpus, RsssuckerFilter(), id='us_politics_test')

    @staticmethod
    def usPoliticsCorpus_test_dedup():
        '''
        Fixed id set of id created from test
         by applying levenshtein clustering and agenda dedup.
        '''
        f = os.path.join(CorpusFactory.res,
                         'dedupIds_us_politics_test_dedup_lev_clusterer_agenda_dedup.pickle')
        idSet = pickle.load(open(f, 'rb'))
        dbase = 'rsssucker_topus1_13042015'
        return IdsetCorpus(dbase, idSet, id='us_politics_test_dedup')

    @staticmethod
    def worldNewsCorpus():
        f = os.path.join(CorpusFactory.res, 'world_news_feeds.txt')
        corpus = FeedsetCorpus('rsssucker_topus1_13042015', Feedlist(f))
        dupfilter = DuplicateTextFilter(RsssuckerCorpus('rsssucker_topus1_13042015'))
        return MultiFilteredCorpus(corpus, [RsssuckerFilter(), dupfilter], id='world_news')

    @staticmethod
    def usNewsCorpus_test():
        f = os.path.join(CorpusFactory.res, 'us_news_feeds.txt')
        corpus = FeedsetCorpus('rsssucker_topus1_13042015_test', Feedlist(f))
        dupfilter = DuplicateTextFilter(RsssuckerCorpus('rsssucker_topus1_13042015_test'))
        return MultiFilteredCorpus(corpus, [RsssuckerFilter(), dupfilter], id='us_news_test')

    @staticmethod
    def usPoliticsCorpus_test_old():
        f = os.path.join(CorpusFactory.res, 'us_politics_feeds.txt')
        corpus = FeedsetCorpus('rsssucker_topus1_13042015_test', Feedlist(f))
        dupfilter = DuplicateTextFilter(RsssuckerCorpus('rsssucker_topus1_13042015_test'))
        return MultiFilteredCorpus(corpus, [RsssuckerFilter(), dupfilter], id='us_politics_test_old')

    @staticmethod
    def worldNewsCorpus_test():
        f = os.path.join(CorpusFactory.res, 'world_news_feeds.txt')
        corpus = FeedsetCorpus('rsssucker_topus1_13042015_test', Feedlist(f))
        dupfilter = DuplicateTextFilter(RsssuckerCorpus('rsssucker_topus1_13042015_test'))
        return MultiFilteredCorpus(corpus, [RsssuckerFilter(), dupfilter], id='world_news_test')