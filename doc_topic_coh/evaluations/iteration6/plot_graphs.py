from matplotlib import pyplot as plt
import pickle
from os import path

graphs2Test = [
    ('eval_graph_params_corpus_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'GRAPH-CNT'),
    ('eval_graph_params_world_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'GRAPH-EMBD'),
    ('eval_distance_params_corpus_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'DISTANCE-CNT'),
    ('eval_distance_params_world_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'DISTANCE-EMBD'),
    ('eval_density_params_corpus_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'DENSITY-CNT'),
    ('eval_density_params_world_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'DENSITY-EMBD'),
]

graphs2TestCro = [
    ('eval_graph_params_corpus_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'GRAPH-CNT'),
    ('eval_graph_params_world_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'GRAPH-EMBD'),
    ('eval_distance_params_corpus_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'DISTANCE-CNT'),
    ('eval_distance_params_world_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'DISTANCE-EMBD'),
    ('eval_density_params_corpus_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'DENSITY-CNT'),
    ('eval_density_params_world_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'DENSITY-EMBD'),
]

from doc_topic_coh.evaluations.iteration6.doc_based_coherence import expFolder, test
from doc_topic_coh.evaluations.iteration5.plot_graphs import algoClassBoxplots
from doc_topic_coh.evaluations.iteration6.best_models import docCohBaseline
from doc_topic_coh.evaluations.iteration6.croelect_topics import model123Topics, alltopics
    #bestParamsDocCroelectIter
bestParamsDocCroelectIter = None
from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer as DCS
from math import log

def bestDocCohParams(reduced=False):
    # params from doc_topic_coh.evaluations.iteration6.best_models.bestParamsDoc
    from pytopia.measure.topic_distance import cosine, l1, l2
    params = [
        # graph-cnt
        {'distance': cosine, 'weighted': False, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'tf-idf',
         'threshold': 50, 'weightFilter': [0, 0.92056], 'type': 'graph'},
        # dist-cnt
        {'distance': cosine, 'center': 'mean', 'vectors': 'probability', 'exp': 1.0,
         'threshold': 50, 'type': 'variance'},
        # dens-cnt
        {'center': 'mean', 'vectors': 'tf-idf', 'covariance': 'spherical',
         'dimReduce': None, 'threshold': 50, 'type': 'density'},
        # graph-emb
        {'distance': l1, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering',
         'vectors': 'glove-avg', 'threshold': 25, 'weightFilter': [0, 14.31248], 'type': 'graph'},
        # dist-emb
        {'distance': cosine, 'center': 'mean', 'vectors': 'glove',
         'exp': 1.0, 'threshold': 25, 'type': 'variance'},
        # dens-emb
        {'center': 'mean', 'vectors': 'word2vec-avg', 'covariance': 'spherical',
         'dimReduce': 10, 'threshold': 25, 'type': 'density'}
    ]
    if reduced:
        params = params[:3]
        params.append(docCohBaseline)
    return params

def plotAucCurves(measureParams, ltopics, axLabels=None,
                  posClass=['theme', 'theme_noise'], grid=None,
                  type='auc', repeat=None, baseline=None):
    from matplotlib import pyplot as plt
    from doc_topic_coh.evaluations.tools import labelMatch
    from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
    from sklearn.metrics import roc_curve, precision_recall_curve
    # create measures from params
    cacheFolder = path.join(expFolder, 'function_cache')
    measures = []
    for p in measureParams:
        p['cache'] = cacheFolder
        measures.append(DocCoherenceScorer(**p)())
    if baseline is not None:
        baseline = DocCoherenceScorer(**baseline)()
    # init grid and plot
    if not grid: grid = (len(measures), 1)
    row, col = 0, 0
    fig, axes = plt.subplots(grid[0], grid[1])
    # render plots
    print len(measures)
    if repeat is not None:
        m = measures[repeat]
        cohrep = [ m(t) for t, tlabel in ltopics ]
        clasrep = [ labelMatch(tlabel, posClass) for t, tlabel in ltopics ]
    if baseline is not None:
        cohbase = [ baseline(t) for t, tlabel in ltopics ]
        clasbase = [ labelMatch(tlabel, posClass) for t, tlabel in ltopics ]
    for i, m in enumerate(measures):
        print row, col
        ax = axes[row, col]
        ax.yaxis.grid(True); ax.xaxis.grid(True)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        cohvals = [ m(t) for t, tlabel in ltopics ]
        classes = [ labelMatch(tlabel, posClass) for t, tlabel in ltopics ]
        pltpar = { 'linewidth': 2 }
        # ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
        #                 labelright='off', labelbottom='off')
        bott, top, left, right = 'off', 'off', 'off', 'off'
        # if col > 0 and row < grid[0]-1:
        #     ax.tick_params(axis='both', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        if col == 0: left = 'on'
        if col == grid[1]-1: right = 'on'
        if row == 0: top = 'on'
        if row == grid[0]-1: bott = 'on'
        ax.tick_params(axis='both',
                       left='off', bottom=bott, top='off', right=right,
                       labelleft='off', labelbottom=bott, labeltop='off', labelright=right)
        if top == 'on':
            ax.set_xlabel(axLabels[1][col], fontsize=25)
            ax.xaxis.set_label_position('top')
        if left == 'on':
            ax.set_ylabel(axLabels[0][row], fontsize=25)
            ax.yaxis.set_label_position('left')

        if type == 'auc':
            fpr, tpr, thresh = roc_curve(classes, cohvals, pos_label=1)
            print 'tpr: ', ','.join('%.3f'%v for v in tpr)
            print 'fpr: ', ','.join('%.3f'%v for v in fpr)
            ax.plot(fpr, tpr, **pltpar)
            if repeat is not None and repeat != i:
                fpr, tpr, thresh = roc_curve(clasrep, cohrep, pos_label=1)
                ax.plot(fpr, tpr, color='red', linewidth=2, linestyle=':')
            if baseline is not None:
                fpr, tpr, thresh = roc_curve(clasbase, cohbase, pos_label=1)
                ax.plot(fpr, tpr, color='green', linewidth=2, linestyle=':')
        elif type == 'prc':
            prec, rec, thresh = precision_recall_curve(classes, cohvals, pos_label=1)
            ax.plot(rec, prec, **pltpar)
        ax.plot([0, 1], [0, 1], color='red', linewidth=0.5, linestyle='--')
        #for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(20)
        #for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(20)
        col += 1
        if col == grid[1]: col = 0; row += 1
    if axLabels:
        rows, cols = grid
        #for r in range(rows):
        #    axes[r, 0].
    plt.tight_layout(pad=0)
    plt.show()

def applyCoh(cohMeasure, labeledTopics):
    return [ cohMeasure(t) for t, tlabel in labeledTopics ]

def plotMeasureCorrelation(m1, m2, ltopics, logscale=None):
    m1, m2 = DCS(**m1)(), DCS(**m2)()
    coh1, coh2 = applyCoh(m1, ltopics), applyCoh(m2, ltopics)
    if logscale:
        if 'x' in logscale: coh1 = [log(c) for c in coh1]
        if 'y' in logscale: coh2 = [log(c) for c in coh2]
    fig, axes = plt.subplots()
    axes.scatter(coh1, coh2)
    plt.show()

def plotMeasureDist(m, ltopics, logscale=False):
    m = DCS(**m)()
    cohs = applyCoh(m, ltopics)
    if logscale: cohs = [log(c) for c in cohs]
    fig, axes = plt.subplots(1, 2)
    print ','.join(['%g'%c for c in sorted(cohs)])
    axes[0].boxplot(cohs)
    axes[1].hist(cohs, bins=100)
    plt.show()

def bestDocCohMeasuresAucCurves(typ='auc', reduced=False, grid = (2, 3),
                                repeat=None, baseline=None):
    plotAucCurves(bestDocCohParams(reduced), test, grid=grid,
                    #axLabels=(['CNT', 'EMB'], ['GRAPH', 'DISTANCE', 'DENSITY']),
                    axLabels=(['BROJANJE RIJEČI', 'VEKTORIZACIJA RIJEČI'],
                              ['GRAF', 'UDALJENOST', 'GUSTOĆA']),
                    type=typ, repeat=repeat, baseline=baseline)

def bestDocCohMeasuresAucCurvesCro(typ='auc', reduced=False, grid = (2, 3),
                                repeat=None, baseline=None):
    plotAucCurves(bestParamsDocCroelectIter(blines=False), alltopics, grid=grid,
                    axLabels=(['CNT', 'EMB'], ['GRAPH', 'DISTANCE', 'DENSITY']),
                    type=typ, repeat=repeat, baseline=baseline)

def plotCorrelation(dataset='uspol'):
    if dataset == 'uspol':
        meas = bestDocCohParams()
        ltopics = test
    elif dataset == 'croelect':
        meas = bestParamsDocCroelectIter()
        ltopics = alltopics
    plotMeasureCorrelation(meas[0], meas[1], ltopics, logscale='x')

if __name__ == '__main__':
    #algoClassBoxplots(algoClassFiles)
    #algoClassBoxplots(graphs2Dev)
    #algoClassBoxplots(graphs2Test, folder=expFolder, select=None)
    #algoClassBoxplots(graphs2TestCro, folder=expFolder, select=None)
                      #saveFile='/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/doc-based coherence/clanak/doc_topic_coherence_ESWA/figures/doccoh_percategory_auc.pdf')
    #bestDocCohMeasuresAucCurves('auc', True, (2,2), repeat=0, baseline=docCohBaseline)
    bestDocCohMeasuresAucCurves('auc', grid=(2, 3), repeat=0, baseline=docCohBaseline)
    #bestDocCohMeasuresAucCurvesCro('auc', grid=(2, 3), repeat=0, baseline=docCohBaseline)
    #plotCorrelation('uspol')
    #plotCorrelation('croelect')
    #plotMeasureDist(bestDocCohParams()[0], test, logscale=True)
    pass