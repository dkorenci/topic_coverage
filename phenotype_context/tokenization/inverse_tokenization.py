from phenotype_context.compose_context import phenotypeContex

from pytopia.context.ContextResolver import resolve
from pytopia.resource.inverse_tokenization.InverseTokenizer import InverseTokenizerBuilder
from pytopia.resource.loadSave import loadResource

INV_TOK_FOLDER = 'pheno_inverse_tokenizer'

def createInvTokenizer():
    with phenotypeContex():
        txt2tok = resolve('PhenotypeText2Tokens')
        corpus = resolve('phenotype_corpus')
        invTok = InverseTokenizerBuilder(corpus, txt2tok, True)
        invTok.save(INV_TOK_FOLDER)

def printInverseTokenization():
    with phenotypeContex():
        d = resolve('pheno_dict1')
        invTok = loadResource(INV_TOK_FOLDER)
        tokens = sorted([t for t in d])
        for t in tokens:
            words = invTok.allWords(t)
            print '%s: %s' % (t, ', '.join(words) if words else '')

if __name__ == '__main__':
    #createInvTokenizer()
    printInverseTokenization()