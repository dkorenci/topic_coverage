from pytopia.tools.IdComposer import IdComposer

class ModelsetStability(IdComposer):
    '''
    Calculate stability of a set of TopicModels, based on a model-pair scorer.
    '''

    def __init__(self, modelMatch):
        '''
        :param modelMatch: function calculating, for two models, a 'matching score'
                that should correlate with model stability
        '''
        self.modelMatch = modelMatch
        IdComposer.__init__(self)

    def __call__(self, modelset):
        '''
        :param modelset: list-like of TopicModels
        '''
        N = len(modelset)
        score = 0.0
        for i, mi in enumerate(modelset):
            for j in range(i+1, N):
                mj = modelset[j]
                score += self.modelMatch(mi, mj)
        if (N != 1):
            score /= (N*(N-1)/2)
        return score
