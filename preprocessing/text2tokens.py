'''
Composite preprocessors that take text as input and produce a list of tokens.
For example, combination of a tokenizer, stopword remover and stemmer.
'''

from normalization import *
from tokenizers import *
from stopwords import *

from nltk.tokenize import regexp_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

class StemmerTokenizer(object):
    '''
    extract sequences of alphabetic characters, lowercase and stem them
    '''    
    alpha = "[a-zA-Z]+"
    stemmer = PorterStemmer()
    stopwords = set(stopwords.words('english'))

    def tokenize(self, text):
        tokens = regexp_tokenize(text, pattern = StemmerTokenizer.alpha)
        tokens = [ tok.lower() for tok in tokens \
                    if tok.lower() not in StemmerTokenizer.stopwords ]
        # remove empty string from the start and end, if they exist
        return [ StemmerTokenizer.stemmer.stem(tok.lower()) for tok in tokens ]

    def __call__(self, text):        
        return self.tokenize(text)

class RsssuckerTxt2Tokens():
    '''
    Performs tokenization, stopword removal and stemming.
    '''

    def __init__(self, originalWords=False):
        self.normalizer = TokenNormalizer(LemmatizerStemmer())
        self.swremover = RsssuckerSwRemover()
        self.tokenizer = regex_word_tokenizer()
        self.originalWords = originalWords

    @property
    def id(self): return self.__class__.__name__

    def tokenize(self, text):
        if not self.originalWords: return self.__tokenizeNonorig(text)
        else: return self.__tokenizeOrig(text)

    def __tokenizeNonorig(self, text):
        ''' return list of stems '''
        tokens = self.tokenizer.tokenize(text)
        return [ self.normalizer.normalize(tok) for tok in tokens if not self.swremover(tok) ]

    def __tokenizeOrig(self, text):
        ''' return list of (stem, originalWord) pairs '''
        tokens = self.tokenizer.tokenize(text)
        return [ (self.normalizer.normalize(tok), tok)
                    for tok in tokens if not self.swremover(tok) ]

    def __call__(self, text):
        return self.tokenize(text)

class RsssuckerTxt2TokensQuick():
    '''
    apply alphabetic sequence tokenization, english stopword removal,
    and porter stemming and lowercasing.
    '''

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.swremover = RsssuckerSwRemover()
        self.tokenizer = regex_word_tokenizer()

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return [ self.stemmer.stem(tok).lower() for tok in self.swremover.remove(tokens) ]

    def __call__(self, text):
        return self.tokenize(text)