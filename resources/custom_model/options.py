class PriorOptions():
    '''
    options for forming topic priors from a set of words
    '''
    def __init__(self, numTopics=None, numWords=None, defWordProb=None,
                 nondefPrior=None, noncustPrior=None, probMass=0.3, strategy = 'per_word'):
        '''
        :param defWordProb: expected probability for a word in a (topic defining) set
        :param nondefPrior: prior for words in a defined topic that are not in the defining set
        :param noncustPrior: prior for words in a non-customized topic
        :param strategy: prior formation strategy, 'per_word' means assign defWordProb probability to
            each prior word, and 'divide_mass' means divide probability mass equally among words
        '''
        self.numTopics = numTopics; self.numWords = numWords
        self.defWordProb = defWordProb; self.nondefWordPrior = nondefPrior
        self.noncustPrior = noncustPrior
        if strategy != 'per_word' and strategy != 'divide_mass':
            raise Exception('undefined topic customization strategy')
        else: self.strategy = strategy
        self.probMass = probMass

    def __str__(self):
        if self.strategy == 'per_word':
            result = 'PR_%s_dwp%.3f_ndwp%.5f_beta%.3f' \
                        % (self.strategy, self.defWordProb, self.nondefWordPrior, self.noncustPrior)
        elif self.strategy == 'divide_mass':
            result = 'PR_%s_mass%.3f_ndwp%.5f_beta%.3f' \
                        % (self.strategy, self.probMass, self.nondefWordPrior, self.noncustPrior)
        return result