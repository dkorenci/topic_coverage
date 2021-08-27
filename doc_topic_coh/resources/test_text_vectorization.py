from doc_topic_coh.resources import pytopia_context

from pytopia.context.ContextResolver import resolve
from pytopia.resource.word_vec_aggregator.WordVecAggregator import WordVecAggregator

def testWord2Vec():
    corpus = resolve('us_politics')
    txt2tok = resolve('RsssuckerTxt2Tokens')
    w2vec = resolve('word2vec_builder')('GoogleNews-vectors-negative300.bin')
    invTok = resolve('inverse_tokenizer_builder')\
                (corpus='us_politics', text2tokens='RsssuckerTxt2Tokens', lowercase=True)
    text2vec = WordVecAggregator(txt2tok, w2vec, invTok)
    cnt = 10
    for txto in corpus:
        print text2vec(txto)
        cnt -= 1
        if cnt == 0: break

def testProbVectors():
    from pytopia.resource.text_prob_vector.TextProbVectorizer import TextProbVectorizer
    vectorizer = TextProbVectorizer(text2tokens='RsssuckerTxt2Tokens',
                                    dictionary='us_politics_dict')
    textVectors = resolve('corpus_text_vectors_builder')\
                    (vectorizer=vectorizer, corpus='us_politics')
    corpus = resolve('us_politics')
    cnt = 10
    for txto in corpus:
        print textVectors(txto)
        cnt -= 1
        if cnt == 0: break

if __name__ == '__main__':
    #testWord2Vec()
    testProbVectors()