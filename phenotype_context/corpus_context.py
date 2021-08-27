'''
Creation of corpora for "Getting the Agenda Right" (gtar) corpora.
'''

from pytopia.context.Context import Context

from phenotype_context.phenotype_corpus.raw_corpus import getCorpus as getRawCorpus
from phenotype_context.settings import corporaRoot, corpusLabels
from phenotype_context.phenotype_corpus.construct_corpus import  \
    getCorpus as getProjectedCorpus

def phenotypeCorpusContext():
    ctx = Context('phenotype_corpus_context')
    ctx.add(getRawCorpus())
    ctx.add(getProjectedCorpus())
    return ctx

def test():
    from pytopia.testing.validity_checks import checkCorpus
    ctx = phenotypeCorpusContext()
    for id in ctx: checkCorpus(ctx[id])

if __name__ == '__main__':
    test()