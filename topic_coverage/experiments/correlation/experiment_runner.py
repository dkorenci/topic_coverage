'''
Evaluation of coverage measures: supervised matcher-based, CTC, ...
'''

from sklearn.metrics import roc_auc_score

from gtar_context.semantic_topics.construct_model import MODEL_ID as GTAR_REFMODEL
from phenotype_context.phenotype_topics.construct_model import MODEL_DOCS_ID as PHENO_REFMODEL
from pytopia.measure.topic_distance import cosine, l1norm, hellinger
from topic_coverage.experiments.correlation.measure_correlation_utils import *
from topic_coverage.experiments.measure_factory import supervisedTopicMatcher
from topic_coverage.modelbuild.modelset_loading import modelset1Families, modelsetLoad
from topic_coverage.resources.pytopia_context import topicCoverageContext

from topic_coverage.settings import topic_models_folder

ctcCovCache = True
supCovCache = True
stabilCache = None
stabilCovCache = None
bipartiteSeparateCache = False

def refmod(corpus):
    if corpus == 'uspol': return resolve(GTAR_REFMODEL)
    elif corpus == 'pheno': return resolve(PHENO_REFMODEL)

def ctcMeasure(level, corpus, strict, typ):
    from topic_coverage.experiments.measure_factory import ctcModelCoverage, cachedTopic2ModelDist
    if typ == 'cosine': topicDist = cosine
    elif typ == 'l1': topicDist = l1norm
    elif typ == 'hellinger': topicDist = hellinger
    else: topicDist = None
    if level == 'model':
        return ModelCoverageFixref(ctcModelCoverage(strict, topicDist, cached=ctcCovCache),
                                   refmod(corpus))
    elif level == 'topic':
        # todo: ?inverse distance
        return cachedTopic2ModelDist(refmod(corpus), topicDist)

def supMeasure(level, corpus, strict, typ=None):
    from topic_coverage.experiments.measure_factory import supervisedModelCoverage, \
        cachedTopic2ModelMatch, supervisedTopicMatcher
    if level == 'model':
        return ModelCoverageFixref(supervisedModelCoverage(corpus, strict, typ, covCache=supCovCache),
                                   refmod(corpus))
    elif level == 'topic':
        return cachedTopic2ModelMatch(refmod(corpus),
                                      supervisedTopicMatcher(corpus, strict, cached=True, typ=typ))

def doccohMeasure(level, corpus, typ):
    from topic_coverage.experiments.coherence.coherence_factory import \
        topDocCohGraph, topDocCohGraph2, topDocCohDist
    if typ == 'graph': coh = topDocCohGraph(corpus)
    if typ == 'graph2': coh = topDocCohGraph2(corpus)
    elif typ == 'dist': coh = topDocCohDist(corpus)
    if level == 'model': return ModelAggTopicCoh(coh)
    elif level == 'topic': return coh

def wordcohMeasure(level, corpus, typ, params=None):
    from topic_coverage.experiments.coherence.coherence_factory import topWordCoh
    coh = topWordCoh(typ, corpus, params)
    if level == 'model': return ModelAggTopicCoh(coh)
    elif level == 'topic': return coh

def stabilityMeasure(type, corpus, families):
    from topic_coverage.experiments.stability.stability_factory import \
        bipartiteStability, relConceptsetStability, ctcStability
    if type == 'bipartite':
        if bipartiteSeparateCache:
            modeltag = '_'.join(sorted(families))
            separateCache = 'bipstabil_%s_models[%s]' % (corpus, modeltag)
        else: separateCache = False
        return bipartiteStability('word-cosine', stabilCache, separateCache)
    elif type == 'rel.conceptset':
        cache = stabilCovCache if stabilCovCache else stabilCache
        return relConceptsetStability(refmod(corpus),
                                      supervisedTopicMatcher(corpus, True, cached=False),
                                      cache)
    elif type == 'ctc':
        cache = stabilCovCache if stabilCovCache else stabilCache
        return ctcStability(cache)
    else: return None

def numTopicsMeasure():
    ''' Return a callable calculating number of topics, either for a TopicModel or a
     list of models (under assumption all the models have the same number of topic). '''
    def numTopics(o):
        m = o[0] if isinstance(o, list) else o
        return m.numTopics()
    return numTopics

def constructMeasure(label, level, corpus, strict=True, typ=None, params=None, families=None):
    '''
    Main interface for measure construction: model/topic coverage: supervised and CTC,
        coherence: word- and document-based
    :param label: 'ctc', 'sup', 'wordcoh', 'doccoh'
    :param level: 'model' or 'topic'
    :param corpus: 'uspol' or 'pheno'
    :param strict: version modifier for ctc and sup
    :param typ: ctc type ('cosine' or 'l1') or coherence type ('graph', 'dist', 'cp', 'cv', 'ca', 'npmi', 'uci')
    :param families: model families
    :return: callable receiving either a topic model or a topic and returning a number
    '''
    if label == 'ctc': return ctcMeasure(level, corpus, strict, typ)
    elif label == 'sup': return supMeasure(level, corpus, strict, typ)
    elif label == 'doccoh': return doccohMeasure(level, corpus, typ)
    elif label == 'wordcoh': return wordcohMeasure(level, corpus, typ, params)
    elif label == 'stability': return stabilityMeasure(typ, corpus, families)
    elif label == 'numT': return numTopicsMeasure()
    else: raise Exception('unknown measure: %s' % label)

def measureLabel(typ, strict=True, subtype=None):
    lab = '%s_strict[%s]_type[%s]' % (typ, strict, subtype)
    return lab

def modelsetLabel(numModels, folder, families, numT):
    lab = 'families[%s]_numModels[%d]_numT[%s]_folder[%s]' % (families, numModels, numT, folder)
    return lab

def calculateCorrelation(m1='ctc', m2='sup', strict1=False, strict2=True, typ1=None, typ2=None,
                         corpus='uspol', level='model',
                         numModels=10, modelsFolder=topic_models_folder, families='all', numT=[50, 100, 200],
                         params1=None, params2=None, oldModelLoad=True, bootstrap=False):
    '''
    :param m1, m2: measure types
    :param strict1, strict2: True/False, subtypes of CTC and SUP
    :param typ1, typ2: type, for ctc or coherence measures
    :param corpus:
    :param level: 'model', 'model.family' or 'topic'
    :param numModels: number of models loaded per (model family, num topics) combination
    :param modelsFolder: folder with stored models
    :param families: list of model families, or 'all'
    :param numT: list of numbers of topics
    :param params1, params2: additional measure parameters
    :return:
    '''
    if oldModelLoad:
        msets, mctx, _ = modelset1Families(corpus, numModels, modelsFolder, families, numT)
    else:
        msets, mctx, _ = modelsetLoad(corpus, modelsFolder, families, numT, autoseg=True)
    with mctx:
        # construct measures
        m1l, m2l = m1, m2
        clevel = 'model' if level in ['model', 'model.family'] else 'topic'
        m1 = constructMeasure(m1, clevel, corpus, strict1, typ1, params1, families)
        m2 = constructMeasure(m2, clevel, corpus, strict2, typ2, params2, families)
        # construct objects on which correlation will be calculated
        if level == 'model': corrobj = [ m for mset in msets for m in mset ]
        elif level == 'model.family': corrobj = msets
        elif level == 'topic': corrobj = [ t for mset in msets for m in mset for t in m ]
        # define correlation measures set
        measures = [spearmanr, pearsonr]
        if (m2l=='wordcoh'): measures = [spearmanr]
        if (m1l == 'stability' or m2l == 'stability'): measures = [spearmanr]
        auc = (m1l == 'sup' and level == 'topic')
        if auc: measures.append(roc_auc_score)
        # calculate correlations and print results
        res = correlation(m1, m2, corrobj, measures,
                          m1l=='stability', m2l=='stability',
                          bootstrap=bootstrap)
        print 'correlation: %s vs %s ; level = %s ; %d datapoints' % \
              (measureLabel(m1l,strict1,typ1), measureLabel(m2l,strict2,typ2), level, len(corrobj))
        print '    models: ', corpus, modelsetLabel(numModels, modelsFolder, families, numT)
        mm = ''
        for i, meas in enumerate(measures):
            r = res[i]
            if isinstance(r, tuple): r, p = r
            else: r, p = r, None
            ml = meas.__name__
            if p is not None: l = '%s %g, p-val %g' % (ml, r, p)
            else: l = '%s %g' % (ml, r)
            mm += (l+ ' ; ')
        print mm
        #auc = (', auc %g'%res[2]) if auc else ''
        #print 'spearman %g, pearson %g%s ; num. datapoints: %d' % (res[0], res[1], auc, len(corrobj))

def tstCorrelationParams(type='cov', corpus='uspol'):
    ''' call calculateCorrelation with all option combinations. '''
    if type == 'cov':
        modelparams = {'numModels':2 , 'families':['lda'], 'numT':[50, 100]}
        for str1 in [True, False]:
            for str2 in [True, False]:
                for lev in ['topic', 'model', 'model.family']:
                    calculateCorrelation('ctc', 'sup', strict1=str1, strict2=str2, level=lev,
                                         corpus=corpus, **modelparams)
                    print
    elif type == 'coh':
        modelparams = {'numModels':1 , 'families':['lda'], 'numT':[50, 100]}
        for typ1 in ['graph', 'dist']:
            for typ2 in ['cv', 'cp']:
                for lev in ['topic', 'model']:
                    calculateCorrelation('doccoh', 'wordcoh', typ1=typ1, typ2=typ2, level=lev,
                                         corpus=corpus, **modelparams)
                    print

def productionCorrelations(corpus='uspol', type='cov'):
    if type == 'cov':
        # sup.strict vs ctc.nonstrict l model
        # //sup.strict vs ctc.strict
        #calculateCorrelation('ctc', 'sup', strict1=False, strict2=True, level='model', corpus=corpus)
        # sup.nonstrict vs ctc.strict l model
        # //sup.nonstrict vs ctc.nonstrict
        calculateCorrelation('ctc', 'sup', strict1=True, strict2=False, level='model', corpus=corpus)
        # sup.strict vs doc.coh l topic
        # sup.nonstrict vs word.coh l topic
        pass
    if type == 'coh':
        calculateCorrelation('doccoh', 'doccoh', typ1='graph', typ2='graph2', level='topic', corpus=corpus)
        #calculateCorrelation('doccoh', 'wordcoh', typ1='dist', typ2='npmi', level='topic', corpus=corpus)
    pass

def supVsCtcCorrelations(corpus='uspol', levels=['model'],
                         ctcTyp=['cosine', 'l1', 'hellinger'], supTyps=[None],
                         families='all'):
    for f in families:
        for typ1 in ctcTyp:
            for supTyp in supTyps:
                for str2 in [True]:
                    for lev in levels:
                        calculateCorrelation('ctc', 'sup', typ1=typ1,
                                             strict2=str2, typ2=supTyp, level=lev, corpus=corpus,
                                             families=f)
                        print

def supVsCtcCorrAll():
    supVsCtcCorrelations('uspol')
    supVsCtcCorrelations('pheno')
    supVsCtcCorrelations('uspol', ctcTyp=['cosine'], supTyps=['nocos'])
    supVsCtcCorrelations('pheno', ctcTyp=['cosine'], supTyps=['nocos'])

def coverageVsCoherenceCorrelations(corpus='uspol'):
    #cohtypes = { 'doccoh':['graph', 'dist'], 'wordcoh':['cv', 'cp'] }
    cohtypes = {'doccoh': ['graph2', 'dist'], 'wordcoh': ['cv', 'cp', 'npmi']}
    covtypes = {'ctc':['cosine'], 'sup':[None]}
    strict = {'ctc':[False], 'sup':[True, False]}
    for cov in ['ctc', 'sup']:
        for covStr in strict[cov]:
            for covTyp in covtypes[cov]:
                for coh in ['doccoh', 'wordcoh']:
                    for cohtyp in cohtypes[coh]:
                        for lev in ['model', 'topic']:
                            calculateCorrelation(cov, coh, strict1=covStr, typ1=covTyp,
                                                 typ2=cohtyp, level=lev, corpus=corpus)
                            print

def coverageVsNpmiCorrelation(corpus='uspol'):
    for lev in ['topic', 'model']:
        for loc in ['.local', '']:
            for ws in [20, 50]:
                print 'WINDOW SIZE', ws
                calculateCorrelation('sup', 'wordcoh', strict1=True, typ1=None,
                             typ2='npmi'+loc, level=lev, corpus=corpus, params2={'windowSize':ws})

def coherenceLocalVsWikiCorr(measures, corpus='uspol'):
    for measure in measures:
        for lev in ['model', 'topic']:
            calculateCorrelation('wordcoh', 'wordcoh', typ1=measure, typ2=measure+'.local',
                             level=lev, corpus=corpus)

def coherenceAllCorr(measures='all', corpus='uspol'):
    if measures == 'all':
        nm = []
        for m in ['npmi', 'cp', 'cv']:
            for l in ['', '.local']: nm.append(m+l)
        measures = nm
    for i, m1 in enumerate(measures):
        for j in range(i+1, len(measures)):
            m2 = measures[j]
            for lev in ['model', 'topic']:
                calculateCorrelation('wordcoh', 'wordcoh', typ1=m1, typ2=m2,
                             level=lev, corpus=corpus)

def coverageVsWordcohCorrelationsBootstrap(corpus='uspol', bootstrap=20000):
    for corp in [corpus]: # 'uspol'
        for coh in ['npmi', 'cp', 'cv']:
            for l in ['', '.local']:
                for lev in ['topic']:
                    calculateCorrelation('sup', 'wordcoh', corpus=corp,
                              strict1=True, typ2=coh+l, level=lev, bootstrap=bootstrap)
                    print
                    calculateCorrelation('ctc', 'wordcoh', corpus=corp,
                              strict1=False, typ1='cosine', typ2=coh+l, level=lev, bootstrap=bootstrap)
                    print

def coverageVsWordcohCorrelations():
    for corp in ['uspol']: # 'uspol'
        for coh in ['npmi', 'cp', 'cv']:
            for l in ['', '.local']:
                for lev in ['topic', 'model']:
                    calculateCorrelation('sup', 'wordcoh', corpus=corp,
                              strict1=False, typ2=coh+l, level=lev)
                    calculateCorrelation('ctc', 'wordcoh', corpus=corp,
                              strict1=False, typ1='cosine', typ2=coh+l, level=lev)

def supCovVsCdcCorrelationsBootstrap(bootstrap=20000):
    for corpus in ['uspol', 'pheno']:
        for suptyp in [None, 'nocos']:
            calculateCorrelation('ctc', 'sup', strict1=False, strict2=True,
                      typ1='cosine', typ2=suptyp, level='model', corpus=corpus,
                      bootstrap=bootstrap)


def stabilityCovCorrelation(corpus='uspol', type='rel.conceptset', bootstrap=10000):
    calculateCorrelation('sup', 'stability', strict1=True, typ2=type,
                         corpus=corpus, level='model.family', bootstrap=bootstrap)
    # calculateCorrelation('sup', 'stability', strict1=False, typ2=type,
    #                      corpus=corpus, level='model.family')
    calculateCorrelation('ctc', 'stability', strict1=False, typ1='cosine', typ2=type,
                         corpus=corpus, level='model.family', bootstrap=bootstrap)

def stabilityCorrelation(corpus='uspol'):
    calculateCorrelation('stability', 'stability', typ1='bipartite', typ2='rel.conceptset', #ctc
                         corpus=corpus, level='model.family')

def stabilityPhenoSeparateCorrelations():
    for fam in [['nmf'], ['lda', 'alda', 'pyp']]:
        calculateCorrelation('sup', 'stability', strict1=True, typ2='rel.conceptset',
                                corpus='pheno', level='model.family', families=fam)
        calculateCorrelation('stability', 'stability', typ1='bipartite', typ2='rel.conceptset',
                                corpus='pheno', level='model.family', families=fam)

def numTopicsCorrelations():
    for corp in ['pheno', 'uspol']:
        print '********* CORPUS: %s ********' % corp
        for meas, typ in [
                            ('sup', None), ('ctc', 'cosine'),
                            #('wordcoh', 'npmi'), ('wordcoh', 'cp'), ('wordcoh', 'cv'),
                            ('wordcoh', 'npmi.local'), ('wordcoh', 'cp.local'), ('wordcoh', 'cv.local'),
                            ('stability', 'bipartite'), ('stability', 'rel.conceptset')
                          ]:
            if meas != 'sup':
                print 'Num.Topics vs %s_%s' % (meas, typ)
                lev = 'model.family' if meas == 'stability' else 'model'
                calculateCorrelation('numT', m2=meas, typ2=typ, strict2=False, level=lev,
                                     corpus=corp)
            else:
                for strict in [True, False]:
                    print 'Num.Topics vs %s_%s' % (meas, 'strict' if strict else 'nonstrict')
                    calculateCorrelation('numT', m2=meas, typ2=typ, strict2=strict, level='model',
                                         corpus=corp)

def bootstrapCorrelationsCtcSup(numResamplings = 100000):
    for supTyp in [None, 'nocos']:
        for corpus in ['uspol', 'pheno']:
            calculateCorrelation('ctc', 'sup', strict1=False, strict2=True,
                         typ1='cosine', typ2=supTyp, level='model', corpus=corpus,
                         bootstrap=numResamplings)

if __name__ == '__main__':
    with topicCoverageContext():
        # calculateCorrelation('sup', 'ctc', strict1=True, strict2=False, typ2='cosine', numModels=1, level='topic',
        #                       families=['lda'], numT=[50, 100])
        #productionCorrelations('pheno', 'coh')
        #supCovVsCdcCorrelationsBootstrap()
        coverageVsWordcohCorrelationsBootstrap('pheno')
        #stabilityCovCorrelation('uspol', 'rel.conceptset', bootstrap=None)
        #bootstrapCorrelationsCtcSup()
        #supVsCtcCorrelations('pheno', families=['lda', 'alda', 'nmf', 'pyp', 'all'], supTyps=['nocos'])
        #supVsCtcCorrelations('pheno')
        #supVsCtcCorrelations('pheno', ctcTyp=['cosine'], supTyps=['nocos'])
        #coverageVsCoherenceCorrelations('uspol')
        #coverageVsCoherenceCorrelations('pheno')
        #coverageVsNpmiCorrelation('uspol')
        #coherenceLocalVsWikiCorr(measures=['cv'], corpus='uspol')
        #coverageVsWordcohCorrelations()
        # calculateCorrelation('sup', 'wordcoh', corpus='pheno',
        #                       strict1=True, typ2='cv.local', level='model')
        #productionCorrelations('pheno', 'cov')
        #testCorrelationParams('cov', 'pheno')
        #testCorrelationParams('coh', 'pheno')
        #stabilityCovCorrelation('uspol', 'rel.conceptset')
        #stabilityPhenoSeparateCorrelations()
        #stabilityCorrelation('pheno')
        #numTopicsCorrelations()
        #coherenceAllCorr(corpus='pheno')
        pass