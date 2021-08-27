from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sets import Set

class TokenNormalizer():
    '''
    Base class for stemmers, lemmatizers, etc.
    Normalizers transform tokens to canonic form and keep a mapings from
    canonic forms to all the variations.
    '''
    def __init__(self, normFunct, storeVariants = False, lowercase = True):
        '''

        :param normFunct: string -> string function performing normalization
        :param storeVariants: it True store map of normalized form to unnormalized forms
        :param lowercase: it True, lowercasing will be performed by default before normFunct
        '''
        self.normFunct = normFunct; self.lowercase = lowercase
        self.storeVariants = storeVariants
        self.norm2token = {}

    def normalize(self, tok):
        tok = tok.lower() if self.lowercase else tok
        ntok = self.normFunct(tok)
        if self.storeVariants :
            if self.norm2token.has_key(ntok):
                self.norm2token[ntok].add(tok)
            else:
                s = Set(); s.add(tok)
                self.norm2token[ntok] = s
        return ntok

class PorterStemmerFunc():
    stemmer = PorterStemmer()
    def __call__(self, token):
        return PorterStemmerFunc.stemmer.stem(token)

class LemmatizerFunc():
    lemmatizer = WordNetLemmatizer()
    def __call__(self, token):
        return LemmatizerFunc.lemmatizer.lemmatize(token)

class LemmatizerStemmer():
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    def __call__(self, token):
        return LemmatizerStemmer.stemmer.stem(
            LemmatizerStemmer.lemmatizer.lemmatize(token))