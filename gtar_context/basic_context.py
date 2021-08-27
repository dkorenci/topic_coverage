from pytopia.context.Context import Context

from gtar_context.corpus_context import gtarCorpusContext
from gtar_context.dict_context import gtarDictionaryContext
from gtar_context.text2tokens_context import gtarText2TokensContext

def basicContext():
    '''
    Context with basic resource only: corpora, dictionaries, tokenizers
    '''
    ctx = Context('gtar_basic_context')
    ctx.merge(gtarCorpusContext())
    ctx.merge(gtarDictionaryContext())
    ctx.merge(gtarText2TokensContext())
    return ctx
