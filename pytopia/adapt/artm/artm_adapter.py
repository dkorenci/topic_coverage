from pytopia.adapt.artm import setup_logging

from pytopia.resource.loadSave import pickleObject
from pytopia.topic_model.TopicModel import TopicModel
from pytopia.context.ContextResolver import resolve, resolveIds
from pytopia.tools.IdComposer import IdComposer

import artm
import numpy as np
from os import path

class ArtmAdapter(TopicModel, IdComposer):
    '''
    Adapts sklearn NMF model to pytopia TopicModel
    '''

    def __init__(self, corpus, dictionary, text2tokens, T, iter=50,
                 type='smspde', rseed=832691, **options):
        '''
        :param corpus: pytopia corpus
        :param dictionary: pytopia dictionary
        :param txt2tokens: pytopia text2tokens
        :param T: number of topics, ie NMF components
        :param type: label of a supported ARTM variant
        :param options: other model-specific options
        :return:
        '''
        self.corpus, self.dictionary, self.text2tokens = \
            resolveIds(corpus, dictionary, text2tokens)
        self.T, self.iter, self.type, self.rseed = T, iter, type, rseed
        atts = ['corpus', 'dictionary', 'text2tokens', 'T', 'type', 'rseed']
        # add model-specific options as object's attributes
        for k, v in options.iteritems():
            self.__dict__[k] = v
            atts.append(k)
        self._docTopics = None
        IdComposer.__init__(self, attributes=atts)

    ### TopicModel interface methods

    def topicIds(self): return range(self.T)

    def numTopics(self): return self.T

    def topicVector(self, topicId):
        self.__constructTopicMatrix()
        return self._topicMatrix[topicId]

    def corpusTopicVectors(self):
        return self._docTopics

    # requires artm_batch_vectorizer_builder
    def build(self):
        '''
        Builds the ARTM model from the constructor params.
        :return:
        '''
        self.__createBatchVectorizer()
        self.__initModel()
        self.__fitModel()
        self.__constructDoctopicMatrix()

    def __createBatchVectorizer(self):
        self._batchVect = resolve('artm_batch_vectorizer_builder')\
                            (self.corpus, self.dictionary, self.text2tokens)

    def __initModel(self):
        _, dictionary = self._batchVect.resource()
        model = artm.ARTM(num_topics=self.T, dictionary=dictionary, cache_theta=True,
                          num_document_passes=1, seed=self.rseed)
        if self.type == 'decorr':
            model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi'))
            model.regularizers['decorrelator_phi'].tau = self.tau
        elif self.type == 'SB':
            topicNames = model.topic_names
            numSpec = int(self.T * self.specific); numBck = self.T - numSpec
            model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi',
                                                                   topic_names=topicNames[numBck:]))
            model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='smooth_phi',
                                                                   topic_names=topicNames[:numBck]))
            model.regularizers['decorrelator_phi'].tau = self.tauSpec
            model.regularizers['smooth_phi'].tau = self.tauBck
        elif self.type == 'smspde':
            # smoothing, sparsing, and decorrelation, as recommended in the orig. article
            tnames = model.topic_names;
            numSpec = int(self.T * self.specific); numBckgrnd = self.T - numSpec
            # smooth both topic-word and topic-doc for "background" topics
            model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='smooth_phi',
                                                                   topic_names=tnames[:numBckgrnd]))
            model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='smooth_theta',
                                                                     topic_names=tnames[:numBckgrnd]))
            model.regularizers['smooth_phi'].tau = self.tauSmPhi
            model.regularizers['smooth_theta'].tau = self.tauSmTh
            # decorrelate topics for "specific" topics
            model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi',
                                                                   topic_names=tnames[numBckgrnd:]))
            model.regularizers['decorrelator_phi'].tau = self.tauDec
            # sparse both topic-word and topic-doc for "specific" topics
            model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi',
                                                                   topic_names=tnames[numBckgrnd:]))
            model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta',
                                                                     topic_names=tnames[numBckgrnd:]))
            model.regularizers['sparse_phi'].tau = self.tauSpPh
            model.regularizers['sparse_theta'].tau = self.tauSpTh
        else: raise Exception('unknown model type: %s'%self.type)
        self._model = model

    def __fitModel(self):
        batch_vectorizer, _ = self._batchVect.resource()
        for i in range(self.iter):
            self._model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

    def __constructTopicMatrix(self):
        if hasattr(self, '_topicMatrix'): return
        d = resolve(self.dictionary); W = d.maxIndex()+1
        m = np.empty((self.T, W), np.float32)
        phi = self._model.phi_ # token X topicName pandas DataFrame
        remapTokInd = {}
        for w in range(W):
            tok = d.index2token(w)
            if tok in phi.index:
                remapTokInd[w] = phi.index.get_loc(tok)
            else:
                #print 'missing word', tok
                remapTokInd[w] = None
        phim = phi.as_matrix()
        #tnames = self._model.topic_names
        for t in range(self.T):
            #print t, tnames[t]
            topicVec = phim[:, t]
            for w in range(W):
                ri = remapTokInd[w]
                # additional validation of equality between m and phi matrices
                # if ri is not None:
                #     assert phi.at[d.index2token(w), tnames[t]] == phim[ri, t]
                val = topicVec[ri] if ri is not None else 0.0
                m[t, w] = val
        self._topicMatrix = m

    def __constructDoctopicMatrix(self):
        ci = resolve('corpus_index_builder')(self.corpus)
        D = len(ci)
        m = np.empty((D, self.T), np.float32)
        theta = self._model.get_theta() # topicName x DocIndex pandas DataFrame
        thetam = theta.as_matrix()
        for d in range(D):
            m[d, :] = thetam[:,d]
        self._docTopics = m

    def __modelDir(self, folder):
        return path.join(folder, 'artm_model')

    def __docTopicsFile(self, folder): return path.join(folder, 'docTopicMatrix.npy')

    def save(self, folder):
        pickleObject(self, folder)
        self._model.dump_artm_model(self.__modelDir(folder))
        if self._docTopics is not None:
            np.save(self.__docTopicsFile(folder), self._docTopics)

    def load(self, folder):
        self._model = artm.load_artm_model(self.__modelDir(folder))
        if path.exists(self.__docTopicsFile(folder)):
            self._docTopics = np.load(self.__docTopicsFile(folder))
        else: self._docTopics = None

from pytopia.resource.ResourceBuilder import SelfbuildResourceBuilder
ArtmAdapterBuilder = SelfbuildResourceBuilder(ArtmAdapter)