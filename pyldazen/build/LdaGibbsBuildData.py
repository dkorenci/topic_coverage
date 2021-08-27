class LdaGibbsBuildData():
    '''
    Data and hyperparams for building LDA models with gibbs sampling.
    '''
    def __init__(self, T, V, documents, alpha, beta, Tf = 0, fixedTopics=None, docFormat='compact-list'):
        '''
        :param V: number of words, in documents words are represented as integers in [0,...,V-1]
        :param T: number of topics
        :param documents: array-like containing documents
        :param alpha: hyperparam controlling document-topic distribution
        :param beta: hyperparam controlling topic-word distribution:
            single number or array of shape (V,) or matrix of shape (T, V)
        :param Tf: number of fixed topics
        :param fixedTopics: matrix of shape (T',V)
        :param docFormat: if 'compact-list' each document is list-like of (word, numWordsInDoc)
                if 'flat-list' each document is list-like of words
        '''
        self.V, self.T, self.documents = V, T, documents
        self.alpha, self.beta = alpha, beta
        self.Tf, self.fixedTopics = Tf, fixedTopics
        self.docFormat = docFormat
