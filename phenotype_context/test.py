from pytopia.context.GlobalContext import GlobalContext
from pytopia.context.ContextResolver import resolve
from phenotype_context.compose_context import phenotypeContex

from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID
from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID
from phenotype_context.phenotype_topics.construct_model import MODEL_DOCS_ID

def printStats():
    with phenotypeContex():
        corpus = resolve(CORPUS_ID)
        print len(corpus)
        dict = resolve(DICT_ID)
        print len(dict)
        model = resolve(MODEL_DOCS_ID)
        print model

if __name__ == '__main__':
    printStats()