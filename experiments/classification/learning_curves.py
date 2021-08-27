import random
import pickle
import os

import matplotlib.pyplot as plt

from experiments.classification.mappers import *
from agenda.mapping.evaluator import MappingEvaluator
from pymedialab_settings.settings import *


def getLCFile(label):
    return object_store+('learning_curves/results_%s.pickle'%label)

def generateLearningCurveData(trainData, testData, labels, sizes,
                              expLabel, resultsRefresh='skip', avgRuns=10, seed=123456):
    '''
    :param trainData: file with labeled texts, for training
    :param testData: file with labeled texts, for testing
    :param labels: list of labels
    :param sizes: list of sizes for the samples of test set
    :param expLabel: label of experiment, for naming the result file
    :param resultsRefresh: if (partial) results are loaded from file, how to add new evaluations
    :avgRuns: for each sample of some size, number of train/test evaluations to average
    :return:
    '''
    random.seed(seed)
    # load labeled data
    train = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+trainData))
    test = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+testData))
    maxSize = len(train)
    # init results data structure
    resultsFile = getLCFile(expLabel)
    if os.path.exists(resultsFile) :
        results = pickle.load(open(resultsFile, 'rb'))
        resultsExist = True
    else:
        results = {} # map {size -> [results for averaging runs]}
        resultsExist = False
    # create learning curve data
    for trainSize in sizes:
        print 'STARTING RUN FOR TRAIN SIZE %d' % trainSize
        if not resultsExist: results[trainSize] = []
        else:
            if trainSize in results:
                if resultsRefresh == 'clear': results[trainSize] = []
                elif resultsRefresh == 'skip':
                    # skip evaluation for sizes that are already in the saved results
                    if trainSize in results:
                        print 'result for size %d already present, skipping' % trainSize
                        continue
            else:
                results[trainSize] = []
        labelClass = {}
        for i in range(avgRuns):
            print '     averaging rung no. %d' % (i+1)
            trainSample = random.sample(train, trainSize)
            for l in labels:
                trainTexts, trainLabels = getTextsAndLabels(trainSample, l)
                print '         training for label %s' % l
                labelClass[l] = trainClassifierForMapper(trainTexts, trainLabels, 'best')
                print '         done.'
            print '     evaluating the mapper'
            evaluator = MappingEvaluator(SupervisedMapper(labelClass), test)
            evaluator.evaluate(labels)
            results[trainSize].append(evaluator.getResults())
            # clear classifier data
            evaluator = None
            labelClass = {}
        # save resutls
        avgMicroF1 = 0.0
        for res in results[trainSize]: avgMicroF1 += res.f1mAvg
        avgMicroF1 /= len(results[trainSize])
        print 'AVERAGE MICRO F1 FOR SIZE %d IS %.3f' % (trainSize, avgMicroF1)
        pickle.dump(results, open(resultsFile, 'wb'))
        print 'SAVED RESULTS. DONE.'

def loadResults(label):
    return pickle.load(open(getLCFile(label), 'rb'))

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
def plotResults(results, score='f1', label=None, prnt=True):
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(4, 2)
    sizes = sorted([sz for sz in results])
    values = []
    for sz in sizes:
        val = 0.0
        for res in results[sz]:
            if label is None:
                if score == 'f1': val += res.f1mAvg
            else:
                labelData = res.labelEval[label]
                if score == 'f1': val += labelData.f1

        val /= len(results[sz])
        if prnt: print '%.3f' % val
        values.append(val)
    sizes[8] = 1600
    axes.plot(sizes, values, linestyle='-', linewidth=0.8, marker='o', color = 'darkblue', markersize = 4)
    #plt.tick_params(axis='x', which='both', bottom='off', top='off')
    #plt.tick_params(axis='y', which='both', left='off', right='off')
    tickf, labf = 7, 8
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(tickf)
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(tickf)
    plt.xlabel('num. of documents', fontsize=labf)
    plt.ylabel('micro F1-score', fontsize=labf)
    plt.tight_layout(pad=0)
    fig.savefig('/home/damir/Dropbox/science/publikacije/papers/agenda_CIKM2015/figures/lc.eps', bbox_inches='tight')
    plt.show()

def plotLearningCurves():
    plotResults(loadResults('lc_alllabels_gran_5AvgRuns'))

def learningCurves():
    # generateLearningCurveData(labeled_files.devel_all_file, labeled_files.test_file,
    #     labels=['lgbt rights','police brutality'],#, 'chapel hill', 'reproductive rights'],
    #     sizes = [800,1000,1200], expLabel='test2', avgRuns=3)
    # generateLearningCurveData(labeled_files.devel_all_file, labeled_files.test_file,
    #                           labels=labels_rights_final(), sizes = [800,1000,1200,1400,1593],
    #                           expLabel='lc_alllabels_5AvgRuns', avgRuns=5, resultsRefresh='skip')
    generateLearningCurveData(labeled_files.devel_all_file, labeled_files.test_file,
                              labels=labels_rights_final(), sizes = [800,900,1000,1100,1200,1300,1400,1500,1593],
                              expLabel='lc_alllabels_gran_5AvgRuns', avgRuns=5, resultsRefresh='skip')

if __name__ == '__main__':
    #plotLearningCurves()
    learningCurves()