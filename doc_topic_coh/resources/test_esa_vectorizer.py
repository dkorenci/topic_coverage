from doc_topic_coh.resources import pytopia_context

from pytopia.resource.esa_vectorizer.EsaVectorizer import EsaVectorizer
from pytopia.resource.corpus_text_vectors.CorpusTextVectors import CorpusTextVectors, \
        CorpusTextVectorsBuilder as builder
from pytopia.context.ContextResolver import resolve

def test(corpus = 'us_politics', dict = 'us_politics_dict', txt2tok = 'RsssuckerTxt2Tokens'):
    esa = EsaVectorizer(corpus=corpus, dictionary=dict, text2tokens=txt2tok)
    c = resolve(corpus)
    cnt = 10
    for txto in c:
        print txto.id
        v = esa(txto.id)
        print 'done'
        cnt -= 1
        if not cnt: break

def testEsaCorpusMatrix(corpus = 'us_politics', dict = 'us_politics_dict', txt2tok = 'RsssuckerTxt2Tokens'):
    esa = EsaVectorizer(corpus=corpus, dictionary=dict, text2tokens=txt2tok, javaVectors=False)
    builder(esa)

if __name__ == '__main__':
    #test()
    testEsaCorpusMatrix()
