from topic_coverage.resources import pytopia_context
from pytopia.context.ContextResolver import resolve

def corpusStats(corpus):
    c = resolve(corpus)
    print len(c)

def modelStats(model):
    m = resolve(model)
    print m.numTopics()

def testNormalization(corpus):
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from pytopia.nlp.text2tokens.regexp import wordTokenizer
    wtok = wordTokenizer()
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    corpus = resolve(corpus)
    allwords = set()
    for i, txto in enumerate(corpus):
        toks = wtok(txto.text)
        allwords.update(set(toks))
    diff = 0
    for t in allwords:
        s = stemmer.stem(t)
        ls = stemmer.stem(lemmatizer.lemmatize(t))
        if s != ls:
            diff += 1
            print 'token: %s ; stem: %s ; lemstem: %s ; lem: %s' % (t, s, ls, lemmatizer.lemmatize(t))
    print 'num.differences: %d' % diff

if __name__ == '__main__':
    #corpusStats('us_politics')
    #corpusStats('us_politics_dedup')
    #corpusStats('pheno_corpus1')
    modelStats('pheno_reftopics')
    #testNormalization('us_politics')