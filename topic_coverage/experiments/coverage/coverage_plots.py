'''
Experiments with varying thresholds for similarity.
'''

from pytopia.measure.avg_nearest_dist import \
    AverageNearestDistance, printAndDetails, TopicCoverDist
from pytopia.measure.topic_distance import cosine as cosineDist

from pytopia.context.ContextResolver import resolve
from pytopia.resource.loadSave import loadResource

from pyutils.stat_utils.utils import Stats

from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS
from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID as PHENO_DICT
from phenotype_context.phenotype_topics.construct_model import MODEL_ID as PHENO_MODEL

from topic_coverage.topicmatch.ctc_matcher import modelDistMatrix

import numpy as np

def coverageForThresholdsBars(corpus, models, thresholds, distMetric=cosineDist,
                              labels=None, plotlabel='coverageForThresholds'):
    '''
    Plot as parallel barcharts coverages of a set of topic models
        for varying topic similarity thresholds.
    Each model is defined as a combination of a model and a number of topics.
    :param corpus: id of the corpus, determines referent target model
    :param models: list of items, each being a topic model (or model id), or a list
            of models/ids, in which case the coverage is averaged
    :param thresholds: list of thresholds for cosine distance
    :param labels: label for each model/modelset
    :return:
    '''
    # setup plot
    from matplotlib import pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1.0]); barw=0.5; cmap = plt.get_cmap('terrain')
    figid = plotlabel; fig.suptitle(plotlabel)
    N = len(thresholds); barx = np.arange(N)
    ax.set_xticklabels(['%g'%t for t in thresholds])
    ax.set_xticks(barx+barw/2)
    ax.yaxis.grid(True)
    # fetch models
    if corpus.startswith('us_politics'): target = resolve('gtar_themes_model')
    elif corpus == PHENO_CORPUS: target = resolve(PHENO_MODEL)
    else: raise Exception('unknown corpus: %s' % corpus)
    # plot
    if not isinstance(models, list): models = [models] # if a single model is given
    numModels = len(models); i = 0
    charts = []; chlab = []
    for i, model in enumerate(models):
        modelset = model if isinstance(model, list) else [model]
        modelset = [resolve(m) for m in modelset]
        covMetrics = [TopicCoverDist(distMetric, th) for th in thresholds]
        means = []
        for dist in covMetrics:
            scores = [ dist(target, m) for m in modelset ]
            stats = Stats(scores)
            means.append(stats.mean)
        ch = ax.bar(barx+barw/numModels*i, means,
                    width=barw/numModels, color=cmap(1.0*i/numModels))
        label = labels[i] if labels else modelset[0].id
        chlab.append('%s'%label)
        charts.append(ch)
    #plt.show()
    ax.legend(charts, chlab, loc=2)
    plt.tight_layout(pad=0)
    plt.savefig(figid+'.pdf')

def coverageForThresholds(target, model, distMetric, thresholds):
    '''
    :param target: referent target modele
    :param model: a topic model (or model id)
    :param thresholds: distance metric thresholds
    :return: ndarray of coverages, one per threshold
    '''
    #todo: factor out model pair distance thresholding ops to ModelDistances(m1, m2) class
    dists = modelDistMatrix(target, model, distMetric)
    closestDist = np.min(dists, axis=1)
    numRef = float(resolve(target).numTopics())
    cov = [None] * len(thresholds)
    for i, th in enumerate(thresholds):
        covered = sum(closestDist <= th)
        cov[i] = covered / numRef
    return np.array(cov)

def coverageForThresholdsAvg(target, models, distMetric, thresholds):
    '''
    :param target: referent target modele
    :param models: a topic model (or model id), or a list of models/ids
    :param thresholds: distance metric thresholds
    :return: ndarray of coverages, one per threshold
    '''
    if not isinstance(models, list): models = [models]
    ctcs = [coverageForThresholds(target, m, distMetric, thresholds) for m in models]
    return np.average(np.array(ctcs), 0)

def modelsetLabel(models):
    model = models[0] if isinstance(models, list) else models
    return model.id if hasattr(model, 'id') else str(model)

def coverageThresholdCurves(target, models, min, max, intervals=100, distMetric=cosineDist,
                              labels=None, plotlabel='ctc', xticks=None):
    '''
    Plot as parallel barcharts coverages of a set of topic models
        for varying topic similarity thresholds.
    Each model is defined as a combination of a model and a number of topics.
    :param target: referent target model
    :param models: list of items, each being a topic model (or model id), or a list
            of models/ids, in which case the coverage is averaged
    :param min, max, intervals: definition of plot points, [min, max] divided into intervals
    :param labels: label for each model/modelset
    :return:
    '''
    # setup plot
    from matplotlib import pyplot as plt
    from scipy.interpolate import interp1d
    fig, ax = plt.subplots()
    plt.xlabel(u'cosine distance threshold', fontsize=10)
    plt.ylabel(u'proportion of reference topics covered', fontsize=12)
    ax.set_ylim([0, 1.0])
    #cmap = plt.get_cmap('seismic')
    #cmap = plt.get_cmap('terrain')
    #cmap = plt.get_cmap('Spectral')
    #colors = ['red', 'blue']; cmap = None;
    #colors = ['tab:orange', 'tab:blue']; cmap = None;
    colors = ['lightcoral', 'darkred', 'red',
              'lightgreen', 'forestgreen', 'lime',
              'skyblue', 'royalblue', 'blue',
              'black'];
    cmap = None;
    figid = plotlabel; #fig.suptitle(plotlabel)
    thresholds = np.linspace(min, max, intervals+1);
    ax.yaxis.grid(True, linestyle='--')
    ax.xaxis.grid(True, linestyle='--')
    ax.set_yticks(np.linspace(0, 1.0, 11))
    if xticks is not None:
        if isinstance(xticks, int): ax.set_xticks(np.linspace(min, max, xticks+1))
        else: ax.set_xticks(xticks)
    for tick in ax.get_xticklabels(): tick.set_fontsize(6)
    # plot
    charts = []; chlab = []; numModels = float(len(models))
    for i, model in enumerate(models):
        covs = coverageForThresholdsAvg(target, model, distMetric, thresholds)
        linInterp = interp1d(thresholds, covs)
        xi = np.linspace(min, max, num=500, endpoint=True)
        label = labels[i] if labels else modelsetLabel(model)
        #ax.plot(thresholds, covs, 'o', color=cmap(i / numModels))
        if cmap is not None: clr = cmap(i / numModels)
        else: clr = colors[i]
        ax.plot(xi, linInterp(xi), linestyle='dashed', linewidth=1,
                color=clr, label=label)
        #print modelsetLabel(model)
        #charts.append(ch)
        #
        #chlab.append('%s'%label)
    #plt.show()
    ax.legend(loc='lower right', prop={'size':7})#(charts, chlab, loc=2)
    #plt.tight_layout(pad=0)
    plt.savefig(figid+'.pdf')


def precisionRecallPlot(target, modelsets, modelCov=None, modelPrec=None, matcher=None, labels=None, plotlabel='prec-rec',
                        groupSize = 3):
    '''
    Plot as parallel barcharts coverages of a set of topic models
        for varying topic similarity thresholds.
    Each model is defined as a combination of a model and a number of topics.
    :param target: referent target model
    :param modelsets: list of items, each being a topic model (or model id), or a list
            of models/ids, in which case the coverage is averaged
    :param labels: label for each model/modelset
    :return:
    '''
    # setup plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.xlabel(u'recall (coverage)', fontsize=12)
    plt.ylabel(u'precision', fontsize=12)
    ax.set_ylim([-0.1, 0.8])
    ax.set_xlim([-0.1, 0.8])
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.5)
    colors = ['lightcoral', 'darkred', 'red',
              'lightgreen', 'forestgreen', 'lime',
              'skyblue', 'royalblue', 'blue',
              'black'];
    lineColors = ['darkred', 'forestgreen', 'royalblue']
    cmap = None;
    figid = plotlabel;
    plt.xticks(np.arange(0.0, 0.8, 0.1))
    plt.yticks(np.arange(0.0, 0.8, 0.1))
    # do plotting
    numModelsets = float(len(modelsets))
    numRefT = float(target.numTopics())
    modelgroup = 0
    precs, recs = [], []
    for i, models in enumerate(modelsets):
        label = labels[i] if labels else modelsetLabel(models)
        if cmap is not None: clr = cmap(i / numModelsets)
        else: clr = colors[i]
        avgRec, avgPrec = 0.0, 0.0
        for m in models:
            if matcher:
                T = float(m.numTopics())
                covered, usedTopics = set(), set()
                for reft in target:
                    for topic in m:
                        if matcher(reft, topic):
                            covered.add(reft.id)
                            usedTopics.add(topic.id)
                            break # only one "hit" per model allowed
                rec, prec = len(covered)/numRefT, len(usedTopics)/T
            else:
                rec = modelCov(target, m)
                prec = modelPrec(target, m)
            avgRec += rec; avgPrec += prec
        avgRec /= len(models); avgPrec /= len(models)
        print(models[0].id)
        print(avgRec, avgPrec)
        ax.plot(avgRec, avgPrec, color=clr, label=label,
                marker='o', fillstyle='full', markersize=6, alpha=0.8)
        precs.append(avgPrec); recs.append(avgRec)
        modelgroup += 1
        if modelgroup % groupSize == 0:
            ax.plot(recs, precs, linestyle='dotted', linewidth=1.2,
                    color= lineColors[modelgroup/groupSize-1])
            precs, recs = [], []
    ax.legend(loc='upper right', prop={'size':7})
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=0)
    plt.savefig(figid+'.pdf')

if __name__ == '__main__':
    pass