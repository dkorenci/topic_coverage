'''
Entry point production runs of stability (correlation) calculations.
'''

from topic_coverage.resources.pytopia_context import topicCoverageContext
from topic_coverage.experiments.correlation import experiment_runner
from topic_coverage.experiments.correlation.experiment_runner import calculateCorrelation

from topic_coverage.settings import topic_models_extended, \
    stability_mock_models, stability_function_cache
from os.path import join

def measureParams(ml):
    '''Convert params to shortcut measure label (cmdline param)
     to measure-defining params of calculateCorrelation() '''
    # ctc bipartite rel.conceptset
    if ml.startswith('st.'): # stability
        pmap = {'ctc':'ctc', 'bip':'bipartite', 'relcss':'rel.conceptset'}
        m = 'stability'
        ml = ml[3:]
        typ = pmap[ml]
        strict = None
    elif ml.startswith('cov.'): # coverage
        ml = ml[4:]
        if (ml == 'ctc'): m, typ, strict = 'ctc', 'cosine', False
        elif (ml == 'sup'): m, typ, strict = 'sup', None, True
    return m, strict, typ

def runStabilUniteTest():
    from topic_coverage.experiments.stability.stability_factory import uniteBipartStabilCaches
    target = '/datafast/topic_coverage/stabil_test/function_cache/stability/unite/'
    sources = [
        '/datafast/topic_coverage/stabil_test/function_cache/stability/unite/stability_bipsep_djurdja/bipstabil_uspol_models[lda]',
        '/datafast/topic_coverage/stabil_test/function_cache/stability/unite/stability_bipsep_djurdja/bipstabil_uspol_models[alda]',
        '/datafast/topic_coverage/stabil_test/function_cache/stability/unite/stability_bipsep_djurdja/bipstabil_uspol_models[nmf]',
        '/datafast/topic_coverage/stabil_test/function_cache/stability/unite/stability_bipsep_djurdja/bipstabil_uspol_models[pyp]',
        '/datafast/topic_coverage/stabil_test/function_cache/stability/unite/stability_bipsep_djurdja/bipstabil_pheno_models[lda]',
        '/datafast/topic_coverage/stabil_test/function_cache/stability/unite/stability_bipsep_djurdja/bipstabil_pheno_models[alda]',
        '/datafast/topic_coverage/stabil_test/function_cache/stability/unite/stability_bipsep_djurdja/bipstabil_pheno_models[nmf]',
        '/datafast/topic_coverage/stabil_test/function_cache/stability/unite/stability_bipsep_djurdja/bipstabil_pheno_models[pyp]',
    ]
    uniteBipartStabilCaches(target, sources)

def runStabilCalc():
    import sys
    print "PARAMS"
    print sys.argv
    print
    modelfold = sys.argv[1]
    covCache = sys.argv[2]
    stabilCache = sys.argv[3]
    corpus = sys.argv[4] # uspol, pheno
    func1 = sys.argv[5] # ctc bipartite rel.conceptset
    func2 = sys.argv[6] #
    modeltyp = sys.argv[7] # "lda alda nmf pyp"
    modeltyp = modeltyp.split()
    if (len(sys.argv) > 8):
        bipartiteSeparateCache = (sys.argv[8] == 'bipsep')
    else: bipartiteSeparateCache = False
    # setup cache
    supCovCache = join(covCache, 'sup_model_coverage')
    ctcCovCache = join(covCache, 'ctc_model_coverage')
    experiment_runner.supCovCache = supCovCache
    experiment_runner.ctcCovCache = ctcCovCache
    experiment_runner.stabilCache = stabilCache
    experiment_runner.bipartiteSeparateCache = bipartiteSeparateCache
    # setup function defining params and run
    m1, str1, typ1 = measureParams(func1)
    m2, str2, typ2 = measureParams(func2)
    calculateCorrelation(m1=m1, m2=m2, strict1=str1, strict2=str2, typ1=typ1, typ2=typ2,
                         level='model.family',corpus=corpus, modelsFolder=modelfold, families=modeltyp,
                          numT=None, oldModelLoad=False)

def runStabilCovCorrelations(corpus, coverage, stability, bootstrap):
    supCovCache = join(stability_function_cache, 'sup_model_coverage')
    ctcCovCache = join(stability_function_cache, 'ctc_model_coverage')
    stabilCache = join(stability_function_cache, 'stability_functions')
    experiment_runner.supCovCache = supCovCache
    experiment_runner.ctcCovCache = ctcCovCache
    experiment_runner.stabilCache = stabilCache
    experiment_runner.stabilCovCache = join(stability_function_cache, 'stability_functions2')
    experiment_runner.bipartiteSeparateCache = False
    m1, str1, typ1 = measureParams(coverage)
    m2, str2, typ2 = measureParams(stability)
    calculateCorrelation(m1=m1, m2=m2, strict1=str1, strict2=str2, typ1=typ1, typ2=typ2,
                         level='model.family',corpus=corpus, modelsFolder=stability_mock_models, families="lda alda nmf pyp",
                          numT=None, oldModelLoad=False, bootstrap=bootstrap)

def runStabilNumtCorrelations(corpus, measure, bootstrap):
    m2, str2, typ2 = measureParams(measure)
    supCovCache = join(stability_function_cache, 'sup_model_coverage')
    ctcCovCache = join(stability_function_cache, 'ctc_model_coverage')
    stabilCache = join(stability_function_cache, 'stability_functions')
    experiment_runner.supCovCache = supCovCache
    experiment_runner.ctcCovCache = ctcCovCache
    experiment_runner.stabilCache = stabilCache
    experiment_runner.bipartiteSeparateCache = False
    calculateCorrelation(m1='numT', m2=m2, strict2=str2, typ2=typ2, level='model.family',
                         corpus=corpus, modelsFolder=stability_mock_models, families="lda alda nmf pyp",
                          numT=None, oldModelLoad=False, bootstrap=bootstrap)

def covStabilNumtCorrelations(corpus):
    runStabilCovCorrelations(corpus, 'cov.sup', 'st.bip', 20000)
    print
    runStabilNumtCorrelations(corpus, 'cov.sup', 20000)
    print
    runStabilCovCorrelations(corpus, 'cov.ctc', 'st.bip', 20000)
    print
    runStabilNumtCorrelations(corpus, 'cov.ctc', 20000)
    print
    runStabilNumtCorrelations(corpus, 'st.bip', 20000)

def stabilStabilCorrelations(corpus):
    runStabilCovCorrelations(corpus, 'st.relcss', 'st.bip', 20000)
    print
    runStabilCovCorrelations(corpus, 'st.ctc', 'st.bip', 20000)

if __name__ == '__main__':
    with topicCoverageContext():
         covStabilNumtCorrelations('uspol')
         #stabilStabilCorrelations('pheno')
