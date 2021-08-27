from topic_coverage.resources.modelsets import *
from topic_coverage.resources.pytopia_context import topicCoverageContext

from topic_coverage.experiments.correlation.experiment_runner import \
    constructMeasure, modelsetLabel, modelset1Families, coverageVsWordcohCorrelationsBootstrap


from pyutils.stat_utils.utils import Stats
from pyutils.stat_utils.plots import basicValueDist

def coherences(coh, objects, prnt=False):
    '''
    Calculate coherence of topics, models or (averages of) lists of models
    :param coh: coherence function accepting an object (topic model or topic), returning a number
    :param objects: list of either objects and/or lists of objects
            (in which case function value is averaged over list elements)
    :param corrFunc: function computing a correlation, or a list of such functions
    :return: calculated coherences
    '''
    import numpy as np
    from scipy.stats import rankdata
    result = []; labels = []
    lab2score = dict()
    for o in objects:
        objlist = o if isinstance(o, list) else [o]
        score = np.average([coh(mod) for mod in objlist])
        result.append(score)
        #TODO - calculate ranks for id-s (pogledaj topNcorr mjere), print rank in addition to score
        lab = objlist[0].id
        lab2score[lab] = score
        labels.append(lab)
    if prnt:
        iscores = [-lab2score[l] for l in labels]
        ranks = rankdata(iscores).astype(int)
        for i, l in enumerate(labels):
            print l
            print '%g %d'%(lab2score[l], ranks[i])
    return result

def cohMeasureLabel(typ, cohtype=None):
    return '%s_type[%s]'%(typ, cohtype)


def calculateCoherence(coh='doccoh', typ='dist', corpus='uspol', level='model', prnt=False, plot=False,
                       numModels=10, modelsFolder=prodModelsBuild, families='all', numT=[50, 100, 200]):
    '''
    :param coh: 'wordcoh' or 'doccoh'
    :param typ: type of coherence
    :param corpus: 'uspol' or 'pheno'
    :param level: 'model', 'model.familiy' or 'topic'
    :param prnt: if true, print all calculated coherences and corresponding obj. ids
    :param numModels: number of models loaded per (model familiy, num topics) combination
    :param modelsFolder: folder with stored models
    :param families: list of model families, or 'all'
    :param numT: list of numbers of topics
    :return:
    '''
    msets, mctx, _ = modelset1Families(corpus, numModels, modelsFolder, families, numT)
    with mctx:
        # construct measures
        cohlabel = coh
        clevel = 'model' if level in ['model', 'model.family'] else 'topic'
        coh = constructMeasure(coh, clevel, corpus, typ=typ)
        # construct objects on which coherence will be calculated
        if level == 'model': corrobj = [ m for mset in msets for m in mset ]
        elif level == 'model.family': corrobj = msets
        elif level == 'topic': corrobj = [ t for mset in msets for m in mset for t in m ]
        # calculate coherences and print summary
        print
        print 'coherences: %s ; level = %s ; %d datapoints' % \
              (cohMeasureLabel(cohlabel, typ), level, len(corrobj))
        print '    models: ', corpus, modelsetLabel(numModels, modelsFolder, families, numT)
        res = coherences(coh, corrobj, prnt=prnt)
        print '    summary stats: ', Stats(res)
        if plot:
            title = '%s_level[%s]_models[%s_%s]' % \
                    (cohMeasureLabel(cohlabel, typ), level, corpus,
                     modelsetLabel(numModels, modelsFolder, families, numT).replace('/','.'))
            basicValueDist(res, title=title, save=True)

def plotAllCoherences():
    levels = ['topic', 'model', 'model.family'] #['model.family'] #['topic', 'model', 'model.family']
    cohtypes = {'doccoh': ['graph', 'dist'], 'wordcoh': ['cv', 'cp', 'npmi']}
    for corpus in ['uspol', 'pheno']:
        for level in levels:
            for coh in ['doccoh', 'wordcoh']:
                for cohtyp in cohtypes[coh]:
                    calculateCoherence(coh, cohtyp, level=level, corpus=corpus, plot=True)

def printModelfamiliyCoherences():
    cohtypes = {'doccoh': ['graph2', 'dist'], 'wordcoh': ['cv', 'cp', 'npmi']}
    for corpus in ['uspol', 'pheno']:
        for coh in ['doccoh', 'wordcoh']:
            for cohtyp in cohtypes[coh]:
                calculateCoherence(coh, cohtyp, level='model.family', corpus=corpus, prnt=True, plot=False)

if __name__ == '__main__':
    with topicCoverageContext():
        #coverageVsWordcohCorrelationsBootstrap('uspol')
        coverageVsWordcohCorrelationsBootstrap('pheno')
        #plotAllCoherences()
        # calculateCoherence('doccoh', 'graph2', level='topic', corpus='pheno', plot=True,
        #                    numModels=1, families=['lda'])
        #printModelfamiliyCoherences()
        #calculateCoherence(numModels=2, plot=True)