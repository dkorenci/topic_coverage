from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve

from pytopia.tools.IdComposer import IdComposer
from pytopia.topic_functions.coherence.doc_matrix_coh_factory import distance_or_matrix_coherence
from pytopia.measure.topic_distance import cosine, kullbackLeibler

class DocCoherenceScorer(IdComposer):
    '''
    Set of parameters for building a topic coherence scorer,
    and methods for building scorers from the parameters.
    '''

    # todo add basic resources as params: corpus, txt2tokens, dict
    def __init__(self, type, vectors=None, threshold=None, topWords=10,
                    corpus='us_politics', text2tokens='RsssuckerTxt2Tokens',
                    dict='us_politics_dict', cache=None,
                    **params):
        '''
        :param type: algorithm type - 'variance', 'avg-dist', 'density', 'graph',
                    'palmetto'
        :param vectors: method for construction of document vectors:
            'tf-idf', 'probability', 'word2vec'
        :param threshold: topic documents selection method
        :param cache: if not None, must cache folder path - then all the
                built scorer functions are wrapped in a CachedFunction
        :param params: algorithm-specific parameters
        :params corpus, text2tokens, dict: ids of basic pytopia resources
        '''
        self.type, self.vectors, self.threshold = type, vectors, threshold
        self.__p = params
        attrs = ['type', 'vectors', 'threshold']
        for k, v in params.iteritems():
            setattr(self, k, v)
            attrs.append(k)
        self.topWords = topWords
        IdComposer.__init__(self, attributes=attrs, class_='Coherence')
        self.corpus, self.text2tokens, self.dictionary = corpus, text2tokens, dict
        self.cache = cache

    def __call__(self):
        self.buildScorer()
        if self.cache is None: return self.__scorer
        else: return self.__cachedScorer()

    def __cachedScorer(self):
        from pytopia.topic_functions.cached_function import CachedFunction
        import os
        from os import path
        if not path.exists(self.cache): os.mkdir(self.cache)
        return CachedFunction(self.__scorer, cacheFolder=self.cache, saveEvery=10)

    def buildScorer(self):
        if self.vectors is not None: self.__docVectorizer()
        if self.type in ['variance', 'avg-dist', 'matrix']: self.__distanceOrMatrixScorer()
        elif self.type == 'density': self.__densityScorer()
        elif self.type == 'graph': self.__graphScorer()
        elif self.type in ['npmi', 'uci', 'umass', 'c_a', 'c_p', 'c_v']:
            self.__palmettoScorer()
        elif self.type == 'tfidf_coherence': self.__tfidfCohScorer()
        elif self.type == 'text_distribution': self.__textDistribScorer()
        elif self.type == 'pairwise_word2vec_destem':  self.__pairwiseW2VScorer()
        elif self.type == 'pairwise_word2vec_wiki':  self.__pairwiseW2VScorerWikiUspolTok()
        elif self.type == 'pairwise_word2vec_uspol': self.__pairwiseW2VScorerUspol()
        else: raise Exception('unknown algorithm type: %s'%self.type)
        return self.__scorer

    # requires 'inverse_tokenizer_builder'
    def __palmettoScorer(self):
        from pytopia.topic_functions.coherence.palmetto_coherence import PalmettoCoherence
        if 'standard' in self.__p and self.__p['standard'] == True:
            # for 'standard' palmetto, original index stores regular words, not stems
            # so inverse tokenization has to be performed
            itb = resolve('inverse_tokenizer_builder')
            itok = itb(self.corpus, self.text2tokens, True)
        else: itok = None
        coh = PalmettoCoherence(self.type, topWords=self.topWords,
                                wordTransform=itok, **self.__p)
        self.__scorer = coh

    # def __palmettoScorerOld(self):
    #     from pytopia.topic_functions.coherence.adapt import WordsStringCohAdapter
    #     from coverexp.coherence.factory import gtarCoherence
    #     def palmettoCoherence(measure, topW):
    #         return WordsStringCohAdapter(coh=gtarCoherence(measure), topW=topW,
    #                                      id='%s[%d]' % (measure, topW))
    #     self.__scorer = palmettoCoherence(self.type, self.topWords)


    def __tfidfCohScorer(self):
        from pytopia.topic_functions.coherence.tfidf_coherence import TfidfCoherence
        self.__scorer = TfidfCoherence(self.topWords)

    def __textDistribScorer(self):
        from pytopia.topic_functions.coherence.document_distribution import DocuDistCoherence
        self.__scorer = DocuDistCoherence(cosine)
        #self.__scorer = DocuDistCoherence(kullbackLeibler)

    def __pairwiseW2VScorer(self):
        from doc_topic_coh.factory.coherences import pairwiseWord2Vec
        self.__scorer = pairwiseWord2Vec(cosine, self.topWords)

    def __pairwiseW2VScorerWikiUspolTok(self):
        from doc_topic_coh.factory.coherences import wikiPairwiseWord2VecUspolTok
        dist = self.distance if hasattr(self, 'distance') else cosine
        self.__scorer = wikiPairwiseWord2VecUspolTok(dist, self.cbow, self.topWords)

    def __pairwiseW2VScorerUspol(self):
        from doc_topic_coh.factory.coherences import uspolPairwiseWord2Vec
        dist = self.distance if hasattr(self, 'distance') else cosine
        self.__scorer = uspolPairwiseWord2Vec(dist, self.cbow, self.vecsize,
                                              self.window, self.topWords)

    def __distanceOrMatrixScorer(self):
        if self.type == 'matrix': self.__p['distance'] = None
        self.__scorer = \
            distance_or_matrix_coherence(self.type, self.threshold,
                                         mapper=self.mapper, mapperIsFactory=self.mapperIsFactory,
                                         **self.__p)

    def __densityScorer(self):
        from pytopia.topic_functions.coherence.density_coherence import GaussCoherence
        from pytopia.topic_functions.document_selectors import TopDocSelector
        from pytopia.topic_functions.topic_elements_score import TopicElementsScore
        select = TopDocSelector(self.threshold)
        score = GaussCoherence(**self.__p)
        self.__scorer = TopicElementsScore(selector=select, score=score,
                                           mapper=self.mapper,
                                           mapperIsFactory=self.mapperIsFactory)

    def __graphScorer(self):
        from pytopia.topic_functions.coherence.doc_matrix_coh_factory import graph_coherence
        self.__scorer = \
            graph_coherence(self.threshold, mapper=self.mapper, mapperIsFactory=self.mapperIsFactory,
                            **self.__p)

    # requires 'corpus_tfidf_builder'
    # requires 'corpus_text_vectors_builder'
    # requires 'word2vec_builder'
    def __docVectorizer(self):
        '''
        Based on self.vectors parameter, create 'mapper' component for TopicElementScore,
        and 'mapperIsFactory' param. Write these params in self.__p dict.
        '''
        if self.vectors == 'tf-idf':
            mapper = resolve('corpus_tfidf_builder')
            mapperIsFactory = True
        elif self.vectors == 'probability':
            #TODO review use-case
            # difference between this and tf-idf is that corpus_tfidf_builder
            # can be used as a factory because it accepts all the basic parameters
            # but prob.vectorizer is composed of top-level textVectors that accepts only corpus
            # containing 2nd-level prob. vectorizer that accepts rest of the params
            # so these params have to be fixed in advance
            from pytopia.resource.text_prob_vector.TextProbVectorizer import TextProbVectorizer
            vectorizer = TextProbVectorizer(text2tokens=self.text2tokens, dictionary=self.dictionary)
            textVectors = resolve('corpus_text_vectors_builder')\
                            (vectorizer=vectorizer, corpus=self.corpus)
            mapper = textVectors
            mapperIsFactory = False
        elif self.vectors.startswith('model'):
            from pytopia.resource.topic_dist_vectorizer.TopicDistVectorizer import TopicDistVectorizer
            if self.vectors in ['model', 'models']:
                modelIds = ['uspolM0', 'uspolM1', 'uspolM2', 'uspolM11', 'uspolM10']
            elif self.vectors == 'models1':
                modelIds = ['models1.1', 'models1.2', 'models1.3', 'models1.4', 'models1.5']
            else: raise Exception('invalid model vectors: %s' % self.vectors)
            mapper = TopicDistVectorizer(self.corpus, modelIds)
            mapperIsFactory = False
        elif self.vectors in ['word2vec', 'glove', 'word2vec-avg', 'glove-avg',
                              'word2vec-cro', 'glove-cro', 'word2vec-cro-avg', 'glove-cro-avg']:
            from pytopia.resource.word_vec_aggregator.WordVecAggregator import WordVecAggregator
            from pytopia.resource.word_vec_aggregator.TfidfWordVecAggreg import TfidfWordVecAggreg
            txt2tok = resolve(self.text2tokens)
            w2vec, glove = self.vectors.startswith('word2vec'), self.vectors.startswith('glove')
            cro = '-cro' in self.vectors
            if w2vec:
                if not cro:
                    embeddings = resolve('word2vec_builder')('GoogleNews-vectors-negative300.bin')
                else: embeddings = resolve('word2vec_builder')('/datafast/word2vec/word2vec.hrwac.cbow.vectors.bin')
            elif glove:
                if not cro:
                    embeddings = resolve('glove_vectors_builder')('/datafast/glove/glove.6B.300d.txt')
                else:
                    embeddings = resolve('glove_vectors_builder')('/datafast/glove/glove.hrwac.300d.txt')
            else:
                raise Exception('unknown vectors: %s' % self.vectors)
            #todo implement without inverse tokenization, with simple
            #whitespace tokenization and stopword removal
            # invTok = resolve('inverse_tokenizer_builder')\
            #     (corpus=self.corpus, text2tokens=self.text2tokens, lowercase=True)
            #text2vec = WordVecAggregator(txt2tok, w2vec, invTok)
            tfidf = self.__p.pop('tfidf', None)
            if not tfidf:
                avg = True if self.vectors.endswith('avg') else None
                if not cro: txt2tok = 'alphanum_gtar_stopword_tokenizer'
                else: txt2tok = 'croelect_alphanum_stopword_tokenizer'
                text2vec = WordVecAggregator(txt2tok, embeddings, None, avg)
            else:
                text2vec = TfidfWordVecAggreg('alphanum_gtar_stopword_tokenizer', embeddings,
                                              'us_politics', 'uspol_dict_notnormalized')
            mapper = resolve('corpus_text_vectors_builder')(vectorizer=text2vec, corpus=self.corpus)
            mapperIsFactory = False
        self.mapper, self.mapperIsFactory = mapper, mapperIsFactory

def scorersFromParams(params):
    '''
    :param params: list of maps representing parameters of DocCoherenceScorer
    :return: list of instantiated DocCoherenceScorers
    '''
    return [ DocCoherenceScorer(**p) for p in params ]