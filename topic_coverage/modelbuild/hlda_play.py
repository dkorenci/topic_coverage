from topic_coverage.resources import pytopia_context

from hlda.sampler import HierarchicalLDA

def corpusDict2Hlda(corpus, text2tok, dict):
    '''
    Convert pytopia corpus and dictionary to hlda input data,
    ie a list of list of word indices (corpus) and a list of words (vocab)
    :return: vocab, corpus
    '''
    from pytopia.context.ContextResolver import resolve
    bow = resolve('bow_corpus_builder')(corpus, text2tok, dict)
    corpus = []
    for doc in bow:
        dlist = []
        for wi, cnt in doc: dlist.extend([wi]*cnt)
        corpus.append(dlist)
    d = resolve(dict)
    vocab = [None]*(d.maxIndex()+1)
    for tok in d: vocab[d[tok]] = tok
    #print '#None in dict: %d' % sum(1 for w in vocab if w is None)
    return vocab, corpus

def runHldaOnPytopiaResources(corpus, text2tok, dict):
    vocab, corpus = corpusDict2Hlda(corpus, text2tok, dict)
    n_samples = 100  # no of iterations for the sampler
    alpha = 10.0  # smoothing over level distributions
    gamma = 1.0  # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
    eta = 0.1  # smoothing over topic-word distributions
    num_levels = 3  # the number of levels in the tree
    display_topics = 10  # the number of iterations between printing a brief summary of the topics so far
    n_words = 20  # the number of most probable words to print for each topic after model estimation
    with_weights = False  # whether to print the words with the weights
    hlda = HierarchicalLDA(corpus, vocab, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)
    hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)

if __name__ == '__main__':
    #corpusDict2Hlda('us_politics_textperline', 'whitespace_tokenizer', 'us_politics_dict')
    runHldaOnPytopiaResources('us_politics_textperline', 'whitespace_tokenizer', 'us_politics_dict')