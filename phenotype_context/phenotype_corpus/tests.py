from pytopia.context.ContextResolver import resolve
from phenotype_context.compose_context import phenotypeContex
from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID

def validRun(subcorpus = 'wikipedia', max=200):
    corpus = resolve(CORPUS_ID)
    i = 0
    for txto in corpus:
        process = subcorpus is None or subcorpus in txto.id
        if process:
            print txto.title
            i += 1
            if i == max-1: break
            print
    print i

def printCorpusIds(max=500):
    corpus = resolve(CORPUS_ID)
    for i, txto in enumerate(corpus):
        if i == max: break
        print txto.id

if __name__ == '__main__':
    with phenotypeContex():
        #validRun(None, -1)
        printCorpusIds()
