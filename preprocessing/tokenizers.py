'''
tokenizers that split text into tokens with no
additional postprocessing such as stemming etc.
'''

from nltk.tokenize import TreebankWordTokenizer

import re

def regex_word_tokenizer():
    'select tokens of alphabetic strings, possibly with one hyphen in the middle'
    #RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    return RegexpTokenizerW('([a-zA-Z]+)(\-[a-zA-Z]+)?')

def penn_tokenizer():
    return TreebankWordTokenizer()

def prefilter_tokenizer():
    'tokenizes out alphanumeric sequences'
    return RegexpTokenizerW('[a-zA-Z0-9]+')

class RegexpTokenizerW():
    'wrapper of RegexpTokenizer that is pickle-able'

    def __init__(self, pattern):
        self.__init_object(pattern)
        self.__init_merge()

    def __init_merge(self):
        self.merge = lambda o: ''.join(o) if isinstance(o, tuple) else o

    def tokenize(self, text):
        if not hasattr(self, 'merge'): self.__init_merge() # for legacy pickled tokenizers
        return [self.merge(o) for o in self.regex.findall(text)]

    def __call__(self, text): return self.tokenize(text)

    def __init_object(self, pattern):
        self.pattern = pattern
        self.regex = re.compile(pattern, flags=re.UNICODE|re.MULTILINE|re.DOTALL)

    def __getstate__(self):
        return self.pattern

    def __setstate__(self, pattern):
        self.__init_object(pattern)