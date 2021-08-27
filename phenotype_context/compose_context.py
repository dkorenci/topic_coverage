from phenotype_context.corpus_context import phenotypeCorpusContext
from phenotype_context.dictionary_context import phenotypeDictContext
from phenotype_context.model_context import phenotypeModelContext
from phenotype_context.tokenization.text2tokens_context import text2TokensContext
from pytopia.context.Context import Context


def phenotypeContex():
    ctx = Context('phenotype_context')
    ctx.merge(phenotypeDictContext())
    ctx.merge(text2TokensContext())
    ctx.merge(phenotypeCorpusContext())
    ctx.merge(phenotypeModelContext())
    return ctx

def printContext():
    from pytopia.context.GlobalContext import GlobalContext
    GlobalContext.set(phenotypeContex())
    print phenotypeContex()

if __name__ == '__main__':
    printContext()