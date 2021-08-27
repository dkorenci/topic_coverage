from pytopia.tools.IdComposer import IdComposer, deduceId

class ClusteringMixednessScorer(IdComposer):
    '''
    Scores a mixedness of a matrix based on a clustering algorithm and a clustering score.
    Mixedness means a measure of how much matrix rows are grouped in two or
    more clusters (bimodal or multimoda distribution) as opposed to
    just one uniform cluster (a unimodal distribution).
    '''

    def __init__(self, clusterer, score, average=1, randomSeed=889, n_jobs=1):
        '''
        :param clusterer: Clustering instance
        :param score: function for scoring clustering of matrix rows,
                accepting matrix and cluster labels
        :param average: run clustering this many times, varying random seed each time
        :param randomSeed: (initial) random seed
        :param n_jobs: number of parallel jobs to use
        '''
        self.clusterer = clusterer
        self.score, self.__score = deduceId(score), score
        self.average = average
        IdComposer.__init__(self)
        self.n_jobs = n_jobs
        self.seed = randomSeed

    def __call__(self, m):
        '''
        :param m: ndarray or scipy sparse matrix
        :return: mixedness score
        '''
        self.clusterer.n_jobs = self.n_jobs
        self.clusterer.fit(m)
        score = 0.0
        for i in range(self.average):
            self.clusterer.random_state = self.seed+i
            score += self.__score(m, self.clusterer.labels_)
        score /= self.average
        return score

