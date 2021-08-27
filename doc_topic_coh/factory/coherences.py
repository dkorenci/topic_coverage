from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve

from pytopia.topic_functions.coherence.wordvectors_dist import WordvecDistCoherence

def pairwiseWord2Vec(dist, topw=10):
    builder = resolve('word2vec_builder')
    word2vec = builder('GoogleNews-vectors-negative300.bin')
    builder = resolve('inverse_tokenizer_builder')
    invTokUspol = builder(corpus='us_politics', text2tokens='RsssuckerTxt2Tokens', lowercase=True)
    return WordvecDistCoherence(word2vector=word2vec, distance=dist,
                                inverseToken=invTokUspol, topWords=topw)

def wikiPairwiseWord2VecUspolTok(dist, cbow, topw=10):
    #TODO: write with uspol wiki
    builder = resolve('word2vec_builder')
    fname = 'word2vec.wiki.cbow%d.en20150602.vectors.bin' % cbow
    word2vec = builder('/datafast/word2vec/%s'%fname)
    return WordvecDistCoherence(word2vector=word2vec, distance=dist, topWords=topw)

def uspolPairwiseWord2Vec(dist, cbow, vecsize, window, topw=10):
    builder = resolve('word2vec_builder')
    fname = 'uspol_word2vec[cbow%d.size%d.window%d].bin' % (cbow, vecsize, window)
    word2vec = builder('/datafast/word2vec/uspol/%s' % fname)
    return WordvecDistCoherence(word2vector=word2vec, distance=dist, topWords=topw)

def uspolWord2VecInvTokensStatic():
    from pytopia.resource.word_vec_aggregator.WordVecAggregator import WordVecAggregator
    txt2tok = resolve('RsssuckerTxt2Tokens')
    w2vec = resolve('word2vec_builder')('GoogleNews-vectors-negative300.bin')
    invTok = resolve('inverse_tokenizer_builder')\
                (corpus='us_politics', text2tokens='RsssuckerTxt2Tokens', lowercase=True)
    text2vec = WordVecAggregator(txt2tok, w2vec, invTok)
    return resolve('corpus_text_vectors_builder')(vectorizer=text2vec,corpus='us_politics')

def uspolProbTextVectorsStatic():
    from pytopia.resource.text_prob_vector.TextProbVectorizer import TextProbVectorizer
    vectorizer = TextProbVectorizer(text2tokens='RsssuckerTxt2Tokens',
                                    dictionary='us_politics_dict')
    textVectors = resolve('corpus_text_vectors_builder') \
        (vectorizer=vectorizer, corpus='us_politics')
    return textVectors

if __name__ == '__main__':
    uspolWord2VecInvTokensStatic()