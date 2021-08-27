from time import time

from experiments.grid_search.options import *
from experiments.grid_search.engine import *
from experiments.grid_search.options import ModelOptions

def gridSearchGenerateModel():
    t = time()
    # uspol1_045_ldamodel_T100_A0.500_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass2_label_seed345556
    options = [ModelOptions(num_topics=100, alpha=0.5, alpha_init=None, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=10)]
    gridSearchParallel(folder='gridsearch_generate_model', corpusId='us_politics',
                       options=options, processes=1, testSize = 2000,
                       seed=112233, propagateSeed=True, shuffleOpts=True, label='', evalPasses=False)
    print 'time: ' + str(time()-t)

def gridSearchNumTopicUsPolAlphaFixed():
    t = time()
    options = gridSearchUsPolNumTopicsOptions(5)
    gridSearchParallel(folder='grid_search_us_politics_numtopics2', corpusId='us_politics',
                       options=options, processes=3, testSize = 2000,
                       seed=718993, propagateSeed=True, shuffleOpts=True, label='numT',
                       evalPasses=False)
    print 'time: ' + str(time()-t)

def gridSearchConvergenceUsPolT100AlphaAuto():
    t = time()
    # uspol2_031_ldamodel_T100_Aauto0.02_Eta0.010_Off10.000_Dec0.500_Chunk1000_Pass2_label_seed345556
    options = [ModelOptions(num_topics=100, alpha='auto', alpha_init=0.02, eta=0.01, offset=10.0,
                            decay=0.5, chunksize=1000, passes=20)]
    gridSearchParallel(folder='grid_search_us_politics_conv_T100_autoalpha', corpusId='us_politics',
                       options=options, processes=1, testSize = 2000,
                       seed=345556, propagateSeed=True, shuffleOpts=True, label='conv', evalPasses=True)
    print 'time: ' + str(time()-t)

def gridSearchConvergenceUsPolT100AlphaFixed2():
    t = time()
    # uspol1_045_ldamodel_T100_A0.500_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass2_label_seed345556
    options = [ModelOptions(num_topics=100, alpha=0.5, alpha_init=0.04, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=10)]
    gridSearchParallel(folder='grid_search_us_politics_conv_T100_fixedalpha2', corpusId='us_politics',
                       options=options, processes=1, testSize = 2000,
                       seed=133890, propagateSeed=True, shuffleOpts=True, label='conv', evalPasses=False)
    print 'time: ' + str(time()-t)

def gridSearchConvergenceUsPolT100AlphaFixed():
    t = time()
    # uspol1_045_ldamodel_T100_A0.500_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass2_label_seed345556
    options = [ModelOptions(num_topics=100, alpha=0.5, alpha_init=0.04, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=20)]
    gridSearchParallel(folder='grid_search_us_politics_conv_T100_fixedalpha', corpusId='us_politics',
                       options=options, processes=1, testSize = 2000,
                       seed=345556, propagateSeed=True, shuffleOpts=True, label='conv', evalPasses=True)
    print 'time: ' + str(time()-t)

def gridSearchConvergenceUsPolT50AlphaAuto():
    t = time()
    # uspol2_018_ldamodel_T50_Aauto0.04_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass2_label_seed345556
    options = [ModelOptions(num_topics=50, alpha='auto', alpha_init=0.04, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=20)]
    gridSearchParallel(folder='grid_search_us_politics_conv_T50_autoalpha', corpusId='us_politics',
                       options=options, processes=1, testSize = 2000,
                       seed=345556, propagateSeed=True, shuffleOpts=True, label='conv', evalPasses=True)
    print 'time: ' + str(time()-t)

def gridSearchConvergenceUsPolT50AlphaFixed():
    t = time()
    options = [ModelOptions(num_topics=50, alpha=1.0, alpha_init=0.01, eta=0.01, offset=1.0,
                     decay=0.5, chunksize=1000, passes=10)]
    gridSearchParallel(folder='grid_search_us_politics_conv_T50_fixedalpha3', corpusId='us_politics',
                       options=options, processes=1, testSize = 2000,
                       seed=8903, propagateSeed=True, shuffleOpts=True, label='', evalPasses=False)
    print 'time: ' + str(time()-t)

def gridSearchConvergenceUsPolT50AlphaFixedConv():
    t = time()
    options = [ModelOptions(num_topics=50, alpha=1.0, alpha_init=0.01, eta=0.01, offset=1.0,
                     decay=0.5, chunksize=1000, passes=30)]
    gridSearchParallel(folder='grid_search_us_politics_conv_T50_fixedalpha', corpusId='us_politics',
                       options=options, processes=1, testSize = 2000,
                       seed=345556, propagateSeed=True, shuffleOpts=True, label='conv', evalPasses=True)
    print 'time: ' + str(time()-t)

def gridSearchPasses_Test():
    t = time()
    options = gridSearchPassesTestOptions()
    gridSearchParallel(folder='grid_search_passes_test', corpusId='us_politics_test',
                       options=options, processes=1, trainSize=500, testSize = 300,
                       shuffleOpts=True, seed=123456, label='passestest', evalPasses=True)
    print 'time: ' + str(time()-t)

def gridSearch_Test():
    t = time()
    options = gridSearchTestOptions()
    gridSearchParallel(folder='grid_search_test', corpusId='us_politics_test',
                       options=options, processes=2, testSize = 500, shuffleOpts=True, label='test')
    print 'time: ' + str(time()-t)

def gridSearch_Worldnews():
    t = time()
    options = gridSearchUsNewsOptions()
    gridSearchParallel(folder='grid_search1_world_news', corpusId='world_news',
                       options=options, processes=3, testSize = 3000,
                       seed=345556, propagateSeed=True, shuffleOpts=True, label='world_news_gs1')
    print 'time: ' + str(time()-t)

def gridSearch_Usnews():
    t = time()
    options = gridSearchUsNewsOptions()
    gridSearchParallel(folder='grid_search1_us_news', corpusId='us_news',
                       options=options, processes=3, testSize = 4000,
                       seed=345556, propagateSeed=True, shuffleOpts=True, label='us_news_gs1')
    print 'time: ' + str(time()-t)

def gridSearch_Politics2():
    t = time()
    options = gridSearchUsPolitics2Options()
    gridSearchParallel(folder='grid_search2_us_politics', corpusId='us_politics',
                       options=options, processes=3, testSize = 2000,
                       seed=345556, propagateSeed=True, shuffleOpts=True, label='us_politics_gs2')
    print 'time: ' + str(time()-t)

def gridSearch_Politics():
    t = time()
    options = gridSearchUsPoliticsOptions()
    gridSearchParallel(folder='grid_search1_us_politics', corpusId='us_politics',
                       options=options, processes=3, testSize = 2000,
                       seed=345556, propagateSeed=True, shuffleOpts=True, label='us_politics_gs1')
    print 'time: ' + str(time()-t)

def gridSearch_TestPolitics():
    t = time()
    options = gridSearchTestSingleOption()
    gridSearchParallel(folder='grid_search_test_politics', corpusId='us_politics',
                       options=options, processes=1, testSize = 2000, shuffleOpts=True, label='test')
    print 'time: ' + str(time()-t)


def gridSearch_variance():
    'test variance of model evaluation metrics for runs with same parameters'
    t = time()
    options = gridSearchVarOptions()
    gridSearchParallel(folder='grid_search_var1', database='rsssucker_topus1_02042015',
                       options=options, processes=3, shuffleOpts=True, label='var')
    print 'time: ' + str(time()-t)

def gridSearch3():
    t = time()
    options = gridSearch3Options()
    gridSearchParallel(folder='grid_search_3', database='rsssucker_topus1_02042015',
                       options=options, processes=2, shuffleOpts=True, label='3')
    print 'time: ' + str(time()-t)

def gridSearch2():
    t = time()
    options = gridSearch2Options()
    gridSearchParallel(folder='grid_search_2', database='rsssucker_topus1_02042015',
                       options=options, processes=2, shuffleOpts=True, label='2')
    print 'time: ' + str(time()-t)