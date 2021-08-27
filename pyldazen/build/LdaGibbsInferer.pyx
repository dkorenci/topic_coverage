#cython: linetrace=True

import numpy as np
from numpy.random import multinomial
from libc.stdlib cimport malloc, free, rand, srand, RAND_MAX

cdef struct Doc:
  long pos # position of first word in word/wordtopic arrays
  long length # number of words
  int *topicCnt # array of topic occurence counts

topicProps = []
topicPropsNorm = []
sampledTopics = []

cdef class LdaGibbsInferer:
    '''
    Infer unknown LDA model parameters based on set of documents, using gibbs sampling.
    '''

    cdef long T, Ta, Tf, randSeed, maxWordIndex
    cdef double alpha, betaNum
    cdef double[:,:] beta, fixedTopics
    cdef bint betaMatrix

    def __cinit__(self, T, alpha, beta, documents, Tf=0, fixedTopics=None,
                        docFormat='bow', maxWordIndex=None, randSeed=1234):
        '''
        '''
        # num of variable, fixed, and all topics
        self.T, self.Tf, self.Ta = T, Tf, T+Tf
        self.alpha, self.fixedTopics = alpha, fixedTopics
        self.maxWordIndex = maxWordIndex
        self.initRandom(randSeed)
        if isinstance(beta, np.ndarray):
            self.betaMatrix = True
            self.__betaNd = beta
            self.beta = beta
            #todo dimension checking
        else:
          self.betaMatrix = False
          self.betaNum = beta
        self.__copyDocuments(documents, docFormat)

    cdef initRandom(self, seed):
      srand(seed) # init C random generator
      np.random.seed(seed)

    def numTopics(self): return self.Ta

    cdef long numWords, numDocs
    cdef long maxWord # maximal value of all word (integers) + 1
    cdef long* w
    cdef Doc* docs
    def __copyDocuments(self, documents, docFormat):
      '''
      Copy document words to internal representation.
      Calculate number of documents and max word.
      '''
      # calc total number of words and maximal word value (words are integers)
      # if there are "holes" in the word integer range, it is input data problem,
      # reindexing to compact the word will not be done and memory will be wasted
      self.numWords, self.numDocs, self.maxWord = 0, 0, 0
      for doc in documents:
        self.numDocs += 1
        if docFormat == 'bow':
          for w, numw in doc:
            self.numWords += numw
            if (self.maxWord < w): self.maxWord = w
        elif docFormat == 'flat-list':
          self.numWords += len(doc)
          for w in doc:
            if (self.maxWord < w): self.maxWord = w
        else: raise Exception('unknown document format')
      # todo: use maxWordIndex instead of maxWord in later calculations
      # print self.maxWord, self.maxWordIndex
      if self.maxWordIndex:
        assert self.maxWord <= self.maxWordIndex
        self.maxWord = self.maxWordIndex + 1
      else: self.maxWord += 1
      # allocate space for words
      self.w = <long*>malloc(self.numWords*sizeof(long))
      self.docs = <Doc*>malloc(self.numDocs*sizeof(Doc))
      # copy words to internal flat arrays
      cdef long dc = 0, wc = 0 # doc and word counters
      cdef long i, nw, docw
      for doc in documents:
        self.docs[dc].pos = wc
        docw = 0
        if docFormat == 'bow':
          for w, numw in doc:
            nw = numw # conversion to C long
            docw += nw
            for i in range(nw):
              self.w[wc] = w; wc += 1
        elif docFormat == 'flat-list':
          docw = len(doc)
          for wrd in doc:
            self.w[wc] = wrd; wc += 1
        self.docs[dc].length = docw
        dc += 1
      print 'num docs %d, num words %d, max word %d' % (self.numDocs, self.numWords, self.maxWord)

    def infer(self):
      '''
      Interface method performing complete inference.
      '''
      self.initData()
      self.runUnsupervisedInference()
      self.__freeMemory()

    cdef initData(self):
      '''
      Allocate and initialize counter variables and other inference variables.
      '''
      self.__initWordtopics()
      self.__initDocTopicCounts()
      self.__initTopicCounts()
      self.__calcBetaSums()

    def startInference(self):
      '''
      Initialize data for user-controlled inference.
      '''
      self.initData()
      self.iterationCounter = 0

    def finishInference(self):
      '''
      Finish user-controlled inference.
      '''
      self.__freeMemory()

    cpdef runInference(self, int steps):
      '''
      User-controlled inference interface, run specified number of gibbs sampling iterations.
      '''
      cdef int i
      for i in range(steps):
        self.iterationCounter += 1
        self.__sampleWordTopics()
        if self.iterationCounter % 10 == 0:
            print '%d gibbs iteration performed' % self.iterationCounter

    cdef bint stopInference
    cdef int iterationCounter
    def runUnsupervisedInference(self):
      '''
      Run inference with gibbs sampling, from initialization to stopping condition.
      '''
      self.stopInference, self.iterationCounter = False, 0
      while not self.stopInference:
        self.iterationCounter += 1
        self.__sampleWordTopics()
        self.checkStop()

    def checkStop(self):
      '''
      Check inference stopping criteria.
      '''
      if self.iterationCounter > 10000: self.stopInference = True

    cdef inline void updateCounters(self, Doc doc, long topic, long word, long value):
      '''
      Update topic-in-doc, global topic, and topic x word token counters by adding value
      '''
      doc.topicCnt[topic] += value
      self.topicCnt[topic] += value
      self.wordTopCnt[word][topic] += value

    cdef __sampleWordTopics(self):
      '''
      Gibbs sample topic assignments for words
      '''
      cdef long di, wi, q, i, newt
      cdef double propsum, b, r, s
      #propsnd = np.ndarray(self.Ta)
      #cdef double[:] props = propsnd
      cdef double *props = <double *>malloc(self.Ta*sizeof(double))
      # run sampling loop
      for di in range(self.numDocs):
        doc = self.docs[di]
        for wi in range(doc.pos, doc.pos+doc.length):
          # calc topic probability proportions
          propsum = 0.0
          v, oldt = self.w[wi], self.z[wi]
          # removing old topic value from position wi
          self.updateCounters(doc, oldt, v, -1)
          # calc proportions for variable topics
          for q in range(self.Ta):
            # topic shuould be "set" q at position wi, ie counters should be incresed
            # instead of increasing (and decreasing) counters, this +1 increase
            # is cancelled with -1 added when calculating sampling proportions, so this -1 is ommited
            # calc proportion for topic q
            props[q] = (self.alpha + doc.topicCnt[q])
            if q < self.T: # variable topics
              if self.betaMatrix: b = self.beta[q][v]
              else: b = self.betaNum
              props[q] *= (self.wordTopCnt[v][q] + b)
              props[q] /= (self.sumBeta[q] + self.topicCnt[q])
            else: # fixed topics
              props[q] *= self.fixedTopics[q-self.T][v]
            #if len(topicProps) < 100000: topicProps.append(props[q])
            propsum += props[q]
          # sample new topic
          r = (<double>rand())/RAND_MAX # sample uniformly from [0,1>
          r *= propsum # normalization
          s = 0.0; newt = -1
          for q in range(self.Ta):
            s += props[q]
            if r < s:
              newt = q
              break
          if newt == -1: newt = self.Ta-1
          # put new topic at position wi
          self.z[wi] = newt
          self.updateCounters(doc, newt, v, 1)
      free(props)

    cdef double [:,:] topics
    def calcTopicMatrix(self):
      '''
      Construct topic - word probabilities from counts and priors
      '''
      cdef long i, j
      cdef double denom
      if not self.betaMatrix:
        ndtop = np.empty((self.T, self.maxWord), dtype=np.float64)
      else:
        ndtop = self.__betaNd.copy()
      self.topics = ndtop
      if not self.betaMatrix: # init topic matrix with beta
        for t in range(self.T):
          for w in range(self.maxWord): self.topics[t][w] = self.betaNum
      for t in range(self.T):
        denom = <double>(self.topicCnt[t]+self.sumBeta[t])
        for w in range(self.maxWord):
          self.topics[t][w] += self.wordTopCnt[w][t]
          self.topics[t][w] /= denom
      return ndtop

    def calcDocTopicMatrix(self):
      cdef long d, t
      doctopic = np.empty((self.numDocs, self.Ta), dtype=np.float64)
      alphaT = self.alpha * self.Ta
      for d in range(self.numDocs):
        for t in range(self.Ta):
            doctopic[d][t] = \
                (<double>(self.docs[d].topicCnt[t]) + self.alpha) / \
                (self.docs[d].length + alphaT)
      return doctopic

    cdef double *sumBeta
    cdef __calcBetaSums(self):
      '''
      Calc sums of word-in-topic priors.
      '''
      self.sumBeta = <double*>malloc(self.T*sizeof(double))
      for ti in range(self.T):
        if self.betaMatrix:
          self.sumBeta[ti] = 0.0
          for wi in range(self.maxWord): self.sumBeta[ti] += self.beta[ti][wi]
        else: self.sumBeta[ti] = self.betaNum * self.maxWord

    cdef int* z
    cdef __initWordtopics(self):
      '''
      Initialize memory for and sample values of word-topics (z).
      '''
      self.z = <int*>malloc(self.numWords*sizeof(int))
      cdef long i
      unif = np.repeat(1.0/self.Ta, self.Ta) # uniform dist over topics
      # todo: can this be done faster, since numpy's multinomial
      # samples vectors of counts, not outcomes directly
      cdef long[:] multi = multinomial(1, unif, self.numWords).argmax(1)
      for i in range(self.numWords):
        self.z[i] = multi[i]

    cdef int* docTopCnt # array of contiguous topic counts
    cdef __initDocTopicCounts(self):
      '''
      Initialize per-document topic occurence counts.
      '''
      self.docTopCnt = <int*>malloc(self.numDocs*self.Ta*sizeof(long))
      cdef long i, j
      cdef Doc* doc
      for i in range(self.numDocs):
        doc = self.docs+i
        doc.topicCnt = self.docTopCnt+i*self.Ta
        for j in range(self.Ta): doc.topicCnt[j] = 0
        for j in range(doc.pos, doc.pos+doc.length):
          doc.topicCnt[self.z[j]] += 1

    cdef long* topicCnt # number of tokens per topic
    cdef long** wordTopCnt # number of word tokens w for topic t (at index [w][t])
    cdef long* wordTopCntBlck
    cdef __initTopicCounts(self):
      '''
      Initialize global topic occurence counts, and per-topic word counts.
      '''
      # alloc and initialize memory
      cdef long i
      self.topicCnt = <long*>malloc(self.Ta*sizeof(long))
      for i in range(self.Ta): self.topicCnt[i] = 0
      self.wordTopCnt = <long**>malloc(self.maxWord*sizeof(long*))
      self.wordTopCntBlck = <long*>malloc(self.maxWord*self.Ta*sizeof(long))
      for i in range(self.maxWord*self.Ta): self.wordTopCntBlck[i] = 0
      for i in range(self.maxWord): self.wordTopCnt[i]=self.wordTopCntBlck+i*self.Ta
      # init counters
      for i in range(self.numWords):
        self.topicCnt[self.z[i]] += 1
        self.wordTopCnt[self.w[i]][self.z[i]] += 1

    cdef __free(self, void **p):
      '''
      Helper that frees pointer and NULLs it if not NULL already.
      '''
      if p[0] != NULL:
        free(p[0])
        p[0] = NULL

    cdef __freeMemory(self):
      self.__free(<void**>&self.docs)
      self.__free(<void**>&self.w)
      self.__free(<void**>&self.z)
      self.__free(<void**>&self.docTopCnt)
      self.__free(<void**>&self.topicCnt)
      self.__free(<void**>&self.wordTopCnt)
      self.__free(<void**>&self.wordTopCntBlck)
      self.__free(<void**>&self.sumBeta)
