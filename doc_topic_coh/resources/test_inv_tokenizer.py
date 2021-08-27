from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve

from pytopia.resource.inverse_tokenization.InverseTokenizer import InverseTokenizer, \
    InverseTokenizerBuilder as builder

def testCreate():
    builder = resolve('inverse_tokenizer_builder')
    invTok = builder(corpus='us_politics', text2tokens='RsssuckerTxt2Tokens', lowercase=True)
    print invTok.id
    print invTok('car'), invTok('citi')

if __name__ == '__main__':
    testCreate()