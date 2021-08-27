'''
Initializer of global context for the pycoverexp project.
To init the context, import module.
'''

from pytopia.context.GlobalContext import GlobalContext
from pytopia.context.Context import Context
from pyutils.file_utils.location import FolderLocation as loc

__contextSet = False

def __initContext():
    global __contextSet
    if __contextSet: return
    GlobalContext.set(topicCoverageContext())
    __contextSet = True

#__initContext()

def topicCoverageContext():
    from gtar_context import gtarContext
    from phenotype_context import phenotypeContex
    from pytopia.nlp.text2tokens.text2tokensContext import basicTokenizersContext
    ctx = Context('topic_coverage_context')
    ctx.merge(gtarContext())
    ctx.merge(phenotypeContex())
    ctx.merge(builderContext())
    ctx.merge(basicTokenizersContext())
    ctx.merge(palmettoContext())
    return ctx

def builderContext():
    from topic_coverage.settings import resource_builder_cache
    from pytopia.resource.builders_context import basicBuildersContext
    ctx = basicBuildersContext(resource_builder_cache)
    return ctx

def palmettoContext():
    '''
    Locations of Palmetto Lucene indexes, for word-based coherence measures.
    :return:
    '''
    from topic_coverage.settings import palmetto_indices
    palmetto_root = loc(palmetto_indices)
    ctx = Context('palmetto_context')
    ctx['wiki_docs'] = palmetto_root('enwiki/lucene_index')
    ctx['wiki_docs_pheno'] = palmetto_root('enwiki_pheno')
    ctx['uspol_palmetto_index'] = palmetto_root('us_politics/windowed')
    ctx['pheno_palmetto_index'] = palmetto_root('pheno')
    return ctx

def testGlobalContext():
    gctx = GlobalContext.get()
    print gctx
    print gctx['corpus_tfidf_builder']

if __name__ == '__main__':
    #testGlobalContext()
    ctx = topicCoverageContext()
    for id in ctx: print id, type(ctx[id])
