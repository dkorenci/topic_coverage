from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve
from coverexp.resources.stem2word import uspolStem2words, uspolStem2wordsNormal

from matplotlib import pyplot as plt

def wordsPerStemDistribution(s2w, stemDict):
    wps = []
    for stem in stemDict:
        if stem in s2w.stem2words:
            wps.append(len(s2w.stem2words[stem]))
        else: print 'missing: ', stem
    print len(wps)
    fig, axes = plt.subplots(2)
    histParams = {'bins': 100}
    axes[0].boxplot(wps)
    axes[1].hist(wps, **histParams)
    plt.show()

def sampleStemByWordnum(s2w, stemDict, wordnum, ss, seed=76652):
    '''
    Sample and display stems with specific number of words.
    :param s2w: mapping of stems to words
    :param ss: sample size
    '''
    from random import sample
    wnStems = [ stem for stem in stemDict
                if stem in s2w.stem2words and
                len(s2w.stem2words[stem]) == wordnum ]
    stems = sample(wnStems, ss)
    for s in stems:
        wordCounts = s2w.stem2words[s].items()
        wordCounts = sorted(wordCounts, key=lambda wc: -wc[1])
        print '%s : %s' % (s, ' '.join('%s:%s'%(w,c) for w,c in wordCounts))


if __name__ == '__main__':
    #wordsPerStemDistribution(uspolStem2wordsNormal(), resolve('us_politics_dict'))
    sampleStemByWordnum(uspolStem2wordsNormal(), resolve('us_politics_dict'),
                        5, 100)