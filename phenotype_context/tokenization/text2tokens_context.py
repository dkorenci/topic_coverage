# -*- coding: utf-8 -*-

from pytopia.context.Context import Context
from pytopia.nlp.text2tokens.regexp import RegexpTokenizer

from nltk.stem.porter import PorterStemmer
import snowballstemmer
from copy import copy

#todo move to pytopia
#todo generalize as text2tokens composed of a tokenizer and normalizer
class PhenotypeText2Tokens():
    '''
    Text 2 tokens processor constructed to match the one in original article:
     porter stemming, hypenation removal, special characters.
    '''

    def __init__(self, swremove=None):
        self.id = self.__class__.__name__
        # include some special characters in the alphabet, for the bio texts
        alphaset=u'[a-zA-Zβφ]'
        alnumset=u'[a-zA-Z0-9βφ]'
        # tokenizer for bio texts that treats as words
        # strings of multiple alphanum seqeunces connected with '-', starting with a letter
        regex = ur'{0}{1}*(?:-{1}+)*'.format(alphaset, alnumset)
        self._tokenizer = RegexpTokenizer('phenotype_tokenizer', regex)
        self._stemmer = snowballstemmer.stemmer('porter')
        self._swremove = swremove

    def __call__(self, text):
        '''
        Whitespace-tokenize and porter-stem string text.
        Remove hyphens within words as is done in original preprocessing.
        '''
        origWords = hasattr(self, 'originalWords') and self.originalWords == True
        text = text.lower()
        tokens = self._tokenizer(text)
        if self._swremove is not None:
            tokens = [t for t in tokens if not self._swremove(t)]
        if origWords: words = copy(tokens)
        tokens = self._stemmer.stemWords(tokens)
        tokens = [t.replace('-', '') for t in tokens]
        if origWords: return [(tok, words[i]) for i, tok in enumerate(tokens)]
        else: return tokens

def text2TokensContext():
    from pytopia.nlp.text2tokens.regexp import whitespaceTokenizer
    ctx = Context('phenotype_text2tokens_context')
    ctx.add(PhenotypeText2Tokens())
    ctx.add(whitespaceTokenizer())
    return ctx

def testTokenizer(text):
    tok = PhenotypeText2Tokens()
    print ','.join(tok(text))

def testSnowball(text):
    import snowballstemmer
    stemmer = snowballstemmer.stemmer('porter')
    #stemmer = PorterStemmer()
    print stemmer.stemWords(text.split())

if __name__ == '__main__':
    #t2t = PhenotypeText2Tokens()
    #print t2t(u'mary had a little lamb')
    #print text2TokensContext()
    #testTokenizer('selenate-reducing carbon-carbon soft-tissue time-consuming')
    #testTokenizer('kidney cry jersey survey')
    #testTokenizer('regular word2 aminocyclopropane-1-carboxylate over-the-counter soft-tissue')
    #testTokenizer('over-the-counter')
    #testSnowball('regular word2 aminocyclopropane-1-carboxylate over-the-counter soft-tissue')
    #testSnowball('kidney cry jersey survey')
    #testTokenizer(u'β-glucan, ctx-φ, β-glucosidas, β-galactosidas, β')
    #testTokenizer('nano-sized')
    #testTokenizer('sized')
    testTokenizer(u'β-galactosidases β-D-galactosidase')