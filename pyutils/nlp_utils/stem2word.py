class Stem2Word():
    '''
    Performs de-stemming, mapping from word 2 stem according to various criteria.
    '''

    def __init__(self, id, lowercase=True):
        self.stem2words = {} # { stem -> {word -> freq} }
        self.topFreq = {} # stem -> freq. of most frequent word
        self.topWord = {} # stem -> most frequent word
        self.id_ = id
        self.lowercase = lowercase

    def id(self): return self.__class__.__name__+('_%s'%self.id_)

    def register(self, stem, word):
        ''' Add (word, stem) data to structures. '''
        if self.lowercase: word = word.lower()
        # increase word count for stem
        if stem not in self.stem2words:
            self.stem2words[stem] = {}
        m = self.stem2words[stem]
        if word not in m: m[word]=1
        else: m[word] += 1
        # update max.freq. word
        if stem not in self.topFreq: self.topFreq[stem] = 0
        if m[word] > self.topFreq[stem]:
            self.topFreq[stem] = m[word]
            self.topWord[stem] = word

    def __call__(self, stem):
        '''return word with highest frequency for the stem'''
        if stem in self.topWord: return self.topWord[stem]
        else: return None

def buildStem2Word(corpus, text2tokens, topWords=None):
    '''
    Build stem to words from corpus with given tokenizer,
    if topWords is not None, process only first topWords.
    '''
    s2wId = 'corpus[%s]_txt2tok[%s]'%(corpus.corpusId(), text2tokens.id)
    s2w = Stem2Word(s2wId)
    print s2w.id()
    cnt = 0
    for txto in corpus:
        for stem, word in text2tokens(txto.text):
            s2w.register(stem, word)
            if topWords:
                cnt += 1
                if cnt == topWords: return s2w
    return s2w

