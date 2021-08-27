from sklearn.metrics import fbeta_score

from doc_topic_coh.evaluations.tools import labelsMatch


def fbeta(beta):
    return lambda true, predicted : fbeta_score(true, predicted, beta=beta)

class ThresholdTopicClassifier():

    def __init__(self, measure, threshold):
        self.measure = measure
        self.threshold = threshold

    def __call__(self, topic):
        r = self.measure(topic)
        if r <= self.threshold: return 1
        else: return 0

def optimizeThreshold(measure, ltopics, label, scoreFunction):
    '''
    :param measure: callable mapping a topic to a number
    :param ltopics: list of labeled topics
    :param label: string or list of strings
    :return:
    '''
    topics = [ t for t, _ in ltopics ]
    realClasses = labelsMatch(ltopics, label)
    measures = [ measure(t) for t in topics ]
    print topics
    print measures
    threshRange = sorted(measures)
    threshRange.extend([min(measures)-1, max(measures)+1])
    bestThreshold = None; maxScore = -1
    for th in threshRange:
        cl = ThresholdTopicClassifier(measure, th)
        calcClasses = [ cl(t) for t in topics ]
        score = scoreFunction(realClasses, calcClasses)
        if score > maxScore:
            maxScore = score
            bestThreshold = th
    return bestThreshold, maxScore

def classificationScore(cl, ltopics, label, scoreFunction):
    '''
    :param cl: classifier
    '''
    realClasses = labelsMatch(ltopics, label)
    calcClasses = [cl(t) for t, _ in ltopics]
    return scoreFunction(realClasses, calcClasses)

from doc_topic_coh.dataset.topic_splits import iter0TrainTestSplit
from pytopia.measure.topic_distance import kullbackLeibler
from pytopia.topic_functions.coherence.tfidf_variance import TfidfVarianceCoherence
from pytopia.topic_functions.coherence.document_distribution import DocuDistCoherence
from doc_topic_coh.evaluations.topic_coherence import palmettoCoherence
def testThresholdClassifier():
    train, test = iter0TrainTestSplit(printStats=False)
    var = TfidfVarianceCoherence()
    ddKL = DocuDistCoherence(kullbackLeibler)
    npmi = palmettoCoherence('npmi', 10)
    measure = npmi
    label = ['theme', 'theme_noise']
    scoreFunction = fbeta(0.1) #precision_score
    th, score = optimizeThreshold(measure, train, label, scoreFunction)
    print 'train %s'%str(scoreFunction), score, type(score)
    cl = ThresholdTopicClassifier(measure, th)
    print 'test  %s'%str(scoreFunction), classificationScore(cl, test, label, scoreFunction)

if __name__ == '__main__':
    testThresholdClassifier()