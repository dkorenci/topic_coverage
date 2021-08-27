import random
import string
import codecs
import pickle
from textwrap import wrap
from time import time

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

from utils.utils import print_text_objects

from corpus.textstream.utils import *
from corpus.filter import *
from corpus.rsssucker import *
from experiments.rsssucker_lda import *
from experiments.grid_search.options import *
from experiments.grid_search.engine import *


def test_tokenizers(seed, file, texts, sampleSize, tokenizers):
    f = codecs.open(file, "w", "utf-8")
    for txt in texts :
        ftext = string.replace(txt, '\n', ' '); ftext = '\n'.join(wrap(txt, 100));
        f.write(ftext+"\n\n")
        for tok in tokenizers :
            tok_list = tok.tokenize(txt)
            tokens =  '\n'.join(wrap(' '.join(tok_list), 100))
            f.write(tok.__class__.__name__+"\n")
            f.write(tokens+"\n\n")
        f.write("****************************************************\n\n")

def getRsssuckerSample(seed, sampleSize, database = 'rsssucker_topus1_27022015'):
    'random sample of texts from the rsssucker database'
    corpus = RsssuckerCorpus(database)
    all_ids = [ id for id in corpus.getIds() ]
    random.seed(seed)
    random.shuffle(all_ids)
    # extract raw texts from (id, Text object) pairs
    texts = [ pair[1].text for pair in corpus.getTexts(all_ids[:sampleSize]) ]
    return texts

def test_ner(seed, file, sampleSize):
    texts = getRsssuckerSample(seed, sampleSize)
    f = codecs.open(file, "w", "utf-8")
    for txt in texts:
        for sentence in nltk.sent_tokenize(txt):
            f.write(sentence+"\n")
            tokens = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            f.write(nltk.ne_chunk(pos_tags, binary=True))
            f.write("\n\n")
        f.write("***************\n\n")

def train_punkt():
    corpus = RsssuckerCorpus('rsssucker_topus1_27022015')
    #sample = getRsssuckerSample(12345, 10)
    texts = [ texto.text for texto in corpus ]
    print 'texts read'
    trainer = PunktTrainer()
    #trainer.INCLUDE_ALL_COLLOCS = True
    cnt = 0
    for text in texts:
        trainer.train(text, finalize=False)
        cnt += 1
        if cnt % 1000 == 0 : print cnt
    trainer.finalize_training()
    tokenizer = PunktSentenceTokenizer(trainer.get_params());
    pickle.dump(tokenizer, open('punkt_rsssucker.pickle', 'wb'))
    for sentence in tokenizer.tokenize(texts[0]): print sentence

def print_filtered_sample(size, seed, database, filtered = True):
    'print a sample of texts filtered out from rsssucker database'
    #txtstream = RsssuckerCorpus('rsssucker_topus1_27022015')
    if filtered:
        corpus = FilteredCorpus(RsssuckerCorpus(database), RsssuckerFilter())
    else:
        corpus = RsssuckerCorpus(database)
    texts = [ txto for txto in corpus ]
    print len(texts)
    random.seed(seed)
    random.shuffle(texts)
    print_text_objects('sample_%s_seed%d_size%d.txt'%(database,seed,size), texts[:size], ordinals=True)

def test_normalizers():
    corpus = FilteredCorpus(RsssuckerCorpus('rsssucker_topus1_27022015'),
                            RsssuckerFilter())
    stemmer = TokenNormalizer(PorterStemmerFunc())
    lemmer = TokenNormalizer(LemmatizerFunc())
    lemmstemmer = TokenNormalizer(LemmatizerStemmer())
    swremove = RsssuckerSwRemover()
    tokenizer = regex_word_tokenizer()
    cnt = 0; max = -1
    for txto in corpus:
        tokens = tokenizer.tokenize(txto.text)
        tokens = [ tok for tok in tokens if not swremove(tok) ]
        for tok in tokens:
            stemmer.normalize(tok)
            lemmer.normalize(tok)
            lemmstemmer.normalize(tok)
        cnt += 1;
        if cnt == max : break

    pickle.dump(stemmer, open('stemmer_rsssucker.pickle', 'wb'))
    pickle.dump(lemmer, open('lemmer_rsssucker.pickle', 'wb'))
    pickle.dump(lemmstemmer, open('lemmstemmer_rsssucker.pickle', 'wb'))

def normalizer_topwords(norm, N = 100):
    'print top N words by number of variants (words normalized to a word)'
    n2t = norm.norm2token
    compare = lambda k1,k2 : -cmp(len(n2t[k1]), len(n2t[k2]))
    keyss = sorted(n2t.keys(), compare)
    for k in keyss[:N]:
        print k + ", " + str(len(n2t[k])) + " : "
        print ' '.join(n2t[k])

def analyze_normalizer(name):
    norm = pickle.load(open(name+'_rsssucker.pickle', 'rb'))
    normalizer_topwords(norm)

def createTopicIndex():
    modelName = 'rsssucker_topus1_27022015_newpreproc_T200alphaauto003_passes2'
    model = loadModel(modelName); text2tokens = RsssuckerTxt2Tokens()
    corpus =  FilteredCorpus(RsssuckerCorpus('rsssucker_topus1_27022015_test'), RsssuckerFilter())
    index = CorpusTopicsIndex(model, corpus, text2tokens)
    pickle.dump(index, open(object_store+'models/'+modelName+'/'+corpus.corpusId()+'_topicIndex.pickle','wb'));

def loadTopicIndex():
    modelName = 'rsssucker_topus1_27022015_newpreproc_T200alphaauto003_passes2'
    corpus =  FilteredCorpus(RsssuckerCorpus('rsssucker_topus1_27022015_test'), RsssuckerFilter())
    return pickle.load(open(object_store+'models/'+modelName+'/'+corpus.corpusId()+'_topicIndex.pickle','rb'));

def testSavedBowsEquality():
    bowPy = loadBowStream_pickle('rsssucker_topus1_27022015')
    bowNumpy = loadBowCorpus('rsssucker_topus1_27022015')
    print bowCorpusesEqual(bowPy, bowNumpy)

def testFeedSetFilter():
    corpus = FeedsetCorpus('rsssucker_topus1_13042015', Feedlist('us_news_feeds.txt'))
    print len(corpus)
    cnt = 0
    for texto in corpus:
        print texto.id, texto.title
        cnt+=1
    print cnt
    for texto in corpus.getTexts([118545,113171,89010,80976]):
        print texto[1].title

def testDuplicateFilter():
    #74138
    corpus = FeedsetCorpus('rsssucker_topus1_13042015', Feedlist('us_news_feeds.txt'))
    #corpus = FeedsetCorpus('rsssucker_topus1_13042015', Feedlist('us_politics_feeds.txt'))
    dupfilter = DuplicateTextFilter(RsssuckerCorpus('rsssucker_topus1_13042015'))
    fcorpus = MultiFilteredCorpus(corpus, [RsssuckerFilter(), dupfilter])
    cnt = 0
    tb = clock()
    for texto in fcorpus:
        #print texto.id, texto.title
        cnt+=1
    print 'time: ' + str(clock()-tb)
    print 'num articles: %d' % cnt
    print 'hash clashes %d  num. hashes %d  fetches %d  duplicates %d' \
          % (dupfilter.hashClashes, len(dupfilter.hash2texts), dupfilter.fetches, dupfilter.duplicates)

from resources.resource_builder import *
def testResourceBuilding():
    #buildDictionaryAndBow('us_politics_test')
    buildDictionaryAndBow('us_politics')

    # opts = ModelOptions(num_topics=50, alpha=1.0, alpha_init=None, eta=0.01,
    #                     offset=1, decay=0.5, chunksize=100, passes=1, label='seed2', seed=226)
    # buildModel('us_politics_test', opts)
