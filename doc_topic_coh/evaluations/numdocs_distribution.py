'''
Analysis of number of documents per topics, for various thresholds.
'''

from doc_topic_coh.dataset.topic_splits import devTestSplit2
from pytopia.topic_functions.document_selectors import TopDocSelector
from doc_topic_coh.evaluations.distance_distribution import statistics

dev, test = devTestSplit2()

def thresholdDistribution(ltopics, th):
    selector = TopDocSelector(th)
    numDoc = [len(selector(t)) for t, _ in ltopics]
    statistics(numDoc, 'numDoc_%.2f'%th)

if __name__ == '__main__':
    thresholdDistribution(dev, 0.15)
