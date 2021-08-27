'''
Creation of pytopia corpora for old croelect corpora.
'''

from pytopia.context.Context import Context
from pytopia.corpus.postgres.feedsucker import *
from pytopia.corpus.filter.duplicate import DuplicateTextFilter
from pytopia.corpus.filter.FilteredCorpus import FilteredCorpus
from pytopia.corpus.filter.textsize import TextsizeFilter
from pytopia.nlp.text2tokens.regexp import alphanumTokenizer

from os import path

thisfolder = path.dirname(__file__)
resfolder = path.abspath(thisfolder)

def getCroelectCorpusContext():
    ctx = Context('croelect_corpus_context')
    f = Feedlist(path.join(resfolder, 'iter0_cronews_feeds.txt'))
    dbase = 'rsssucker_croelect_14012015_iter0filled'
    fsetcorpus = FeedsetCorpus(dbase, f, startDate='2015-09-30 00:00:00',
                                    endDate='2015-12-28 23:59:59')
    dupfilter = DuplicateTextFilter(FeedsuckerCorpus(dbase))
    sizeFilter = TextsizeFilter(40, alphanumTokenizer())
    corpus = FilteredCorpus('iter0_cronews_final', fsetcorpus, [sizeFilter, dupfilter])
    ctx.add(corpus)
    return ctx

def testCorpus(corpusId='iter0_cronews_final'):
    from coverexp.pytopia_devel.tests.corpus_tests import compareCorpora
    from croelect.resources.corpus_factory import CorpusFactory
    cnew = getCroelectCorpusContext()[corpusId]
    cold = CorpusFactory.getCorpus(corpusId)
    compareCorpora(cnew, cold)

if __name__ == '__main__':
    testCorpus()