from pytopia.resource.word2vec.word2vec import Word2VecBuilder
from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve

def test(bid='word2vec_builder', fname='/datafast/word2vec/GoogleNews-vectors-negative300.bin'):
    builder = resolve(bid)
    w2vec = builder(fname)
    print len(w2vec)

def testClosest(bid='word2vec_builder', fname='/datafast/word2vec/GoogleNews-vectors-negative300.bin'):
    builder = resolve(bid)
    w2vec = builder(fname)
    from sys import stdin
    print 'enter a word per line, blank line aborts'
    while True:
        line = stdin.readline()
        line = line.strip()
        if line == '': break
        res = w2vec.closest(line, 10)
        print ' '.join(res)

if __name__ == '__main__':
    #test('glove_vector_builder', '/datafast/glove/glove.6B.300d.txt')
    testClosest('glove_vectors_builder', '/datafast/glove/glove.6B.300d.txt')
    #testClosest()