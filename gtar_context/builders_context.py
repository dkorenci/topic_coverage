'''
Context with resource builders for internal within-package use.
'''

from pytopia.resource.FolderResourceCache import FolderResourceCache
from pyutils.file_utils.location import FolderLocation as loc

from os import path

# determine cache folder for resource builders
thisfolder = path.dirname(__file__)
from gtar_context import settings
if hasattr(settings, 'resource_builder_cache'):
    cacheFolder = loc(settings.resource_builder_cache)
else:
    cacheFolder = loc(path.abspath(path.join(thisfolder, 'resource_builder_cache')))

def corpusIndexBuilder():
    from pytopia.resource.corpus_index.CorpusIndex import CorpusIndexBuilder
    return FolderResourceCache(CorpusIndexBuilder(), cacheFolder('corpus_index'),
                               id = 'corpus_index_builder')

def corpusTfidfBuilder():
    from pytopia.resource.corpus_tfidf.CorpusTfidfIndex import CorpusTfidfBuilder
    return FolderResourceCache(CorpusTfidfBuilder(), cacheFolder('corpus_tfidf_index'),
                               id = 'corpus_tfidf_builder')

def bowCorpusBuilder():
    from pytopia.resource.bow_corpus.bow_corpus import BowCorpusBuilder
    return FolderResourceCache(BowCorpusBuilder(), cacheFolder('bow_corpus'),
                               id = 'bow_corpus_builder')

def buildersContext():
    from pytopia.context.Context import Context
    ctx = Context('gtar_context_resource_builders')
    ctx.add(corpusIndexBuilder())
    ctx.add(corpusTfidfBuilder())
    ctx.add(bowCorpusBuilder())
    return ctx



