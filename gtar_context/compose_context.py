from pytopia.context.Context import Context

from gtar_context.corpus_context import gtarCorpusContext
from gtar_context.dict_context import gtarDictionaryContext
from gtar_context.text2tokens_context import gtarText2TokensContext
from gtar_context.orig_models.orig_model_context import gtarModelsContext
from gtar_context.semantic_topics.construct_model import gtarRefModelsContext

def gtarContext():
    ctx = Context('gtar_context')
    ctx.merge(gtarCorpusContext())
    ctx.merge(gtarDictionaryContext())
    ctx.merge(gtarText2TokensContext())
    ctx.merge(gtarModelsContext())
    ctx.merge(gtarRefModelsContext())
    return ctx
