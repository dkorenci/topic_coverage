from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve
from pytopia.topic_functions.function_composition import FunctionComposition

def uspolInvTokenWord2Vec():
    txt2tok = resolve('RsssuckerTxt2Tokens')
    w2vec = resolve('word2vec_builder')('GoogleNews-vectors-negative300.bin')
    invTok = resolve('inverse_tokenizer_builder')\
                (corpus='us_politics', text2tokens='RsssuckerTxt2Tokens', lowercase=True)
    return FunctionComposition(w2vec, invTok)
