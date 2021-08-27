from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve

import codecs

def corpus2text(corpusId, text2tokens, file='corpus.txt'):
    '''
    Output pytopia corpus to text file, tokenized file per line
    '''
    corpus, txt2tok = resolve(corpusId, text2tokens)
    f = codecs.open(file, 'w', 'utf-8')
    for txto in corpus:
        tokens = txt2tok(txto.text)
        line = u' '.join(tokens)
        f.write(line); f.write('\n')

def uspolCorpus2Text():
    corpus2text('us_politics', 'RsssuckerTxt2Tokens', 'uspol_corpus.txt')

def croelectCorpus2Text():
    from doc_topic_coh.resources.croelect_resources.croelect_resources import corpusId, dictId, text2tokensId
    corpus2text(corpusId, text2tokensId, 'croelect_corpus.txt')

if __name__ == '__main__':
    croelectCorpus2Text()
