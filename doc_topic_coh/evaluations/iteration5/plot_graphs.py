from matplotlib import pyplot as plt
import pickle, numpy as np
from os import path

algoClassFiles = [
    ('top graph corpus.pickle', 'graph corpus'),
    ('top graph world.pickle', 'graph world'),
    ('top distance corpus.pickle', 'dist corpus'),
    ('top distance world.pickle', 'dist world'),
    ('top density corpus.pickle', 'gauss corpus'),
    ('top density world.pickle', 'gauss world'),
]

graphs2Dev = [
    ('eval_graph_params_corpus_vectors_dev_topic_split_[devSize=120]_[seed=9984].pickle', 'graph corpus'),
    ('eval_graph_params_world_vectors_dev_topic_split_[devSize=120]_[seed=9984].pickle', 'graph world'),
    ('eval_distance_params_corpus_vectors_dev_topic_split_[devSize=120]_[seed=9984].pickle', 'distance corpus'),
    ('eval_distance_params_world_vectors_dev_topic_split_[devSize=120]_[seed=9984].pickle', 'distance world'),
    ('eval_density_params_corpus_vectors_dev_topic_split_[devSize=120]_[seed=9984].pickle', 'probability corpus'),
    ('eval_density_params_world_vectors_dev_topic_split_[devSize=120]_[seed=9984].pickle', 'probability world'),
]
graphs2DevSelectThresh = [0.8, 0.75, 0.74, 0.73, 0.74, 0.63]

graphs2Test = [
    ('eval_graph_params_corpus_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'graph corpus'),
    ('eval_graph_params_world_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'graph world'),
    ('eval_distance_params_corpus_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'distance corpus'),
    ('eval_distance_params_world_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'distance world'),
    ('eval_density_params_corpus_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'probability corpus'),
    ('eval_density_params_world_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'probability world'),
]

def algoClassBoxplots(lfiles, folder='/datafast/doc_topic_coherence/experiments/iter5_coherence/',
                      select=None, saveFile=None):
    results = []
    for lf in lfiles:
        r = pickle.load(open(path.join(folder, lf[0]), 'rb'))
        print '%15s num params: %d' % (lf[1], len(r))
        ordered = sorted(r, reverse=True)
        print ' '.join(('%.4f'%v) for v in ordered)
        selected = r[0]; best = ordered[0]
        print 'selected %.4f, best %.4f, diff %.4f' % (selected, best, abs(best-selected))
        results.append(r)
    fig, axes = plt.subplots(1, 1)
    axes.set_ylim([0.6, 0.85])
    axes.boxplot(results, showfliers=False)
    xcoord = range(1, len(results)+1) # x coordinates of boxes
    for i, res in enumerate(results):
        axes.scatter([xcoord[i]] * len(res), res, alpha=0.4)
        # plot the average of the top quartile
        q75 = np.percentile(res, 75)
        resq75 = [r for r in res if r >= q75]
        avgq75 = np.average(resq75)
        axes.plot(xcoord[i], avgq75, mec='blue', marker='o', mew=4, markersize=25, mfc="None")
        # plot marker on the top-dev result
        axes.plot(xcoord[i], res[0], color='r', marker='x', mew=6, markersize=30)
        if select:
            linex = xcoord[i]+0.35
            best = max(res)
            brackw = 0.03
            axes.plot([linex-brackw, linex, linex, linex, linex, linex-brackw ],
                      [select[i], select[i], select[i], best, best, best],
                      'r', linewidth=2)

    #axes.xaxis.tick_top()
    # Set the labels
    labels = [lf[1] for lf in lfiles]
    axes.set_xticklabels(labels, minor=False)
    for tick in axes.xaxis.get_major_ticks(): tick.label.set_fontsize(27)
    for tick in axes.yaxis.get_major_ticks(): tick.label.set_fontsize(27)
    #plt.xticks(rotation=45)
    axes.yaxis.grid(True)
    # Turn off x ticks
    for t in axes.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    plt.tight_layout(pad=0)
    if saveFile: fig.savefig(filename=saveFile, box_inches='tight')
    else: plt.show()

if __name__ == '__main__':
    #algoClassBoxplots(algoClassFiles)
    #algoClassBoxplots(graphs2Dev)
    algoClassBoxplots(graphs2Test, select=None)
                      #saveFile='/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/doc-based coherence/clanak/doc_topic_coherence_ESWA/figures/doccoh_percategory_auc.pdf')