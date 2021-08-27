from pytopia.context.ContextResolver import resolve

import random

def printText(txto):
    ''' Print pytopia text object, in one line '''
    line = ' '.join(txto.text.split())
    print 'Text id: %s, text: %s' % (str(txto.id), line)

def checkCorpus(corpus, numTexts=10, seed=123):
    corpus = resolve(corpus)
    print 'corpus id: %s' % str(corpus.id)
    print 'corpus length: %d' % len(corpus)
    print 'first %d texts' % numTexts
    for i, txto in enumerate(corpus):
        if i == numTexts: break
        printText(txto)
    print 'random text sample'
    random.seed(seed)
    ids = random.sample(corpus.textIds(), 10)
    for txto in corpus.getTexts(ids): printText(txto)

def checkDictionary(d, toPrint=10, seed=123):
    print 'dictionary id: %s' % str(d.id)
    print 'maxIndex: %d' % d.maxIndex()
    print 'first %d key-value pairs' % toPrint
    for i, kv in enumerate(d.iteritems()):
        if i == toPrint: break
        print kv[0], kv[1]
    print 'random %d words' % toPrint
    random.seed(seed)
    idx = random.sample(d.values(), toPrint)
    tokens = [d.index2token(i) for i in idx]
    print ' '.join(tokens)
    print 'random words as bag-of-words'
    print d.tokens2bow(tokens)
