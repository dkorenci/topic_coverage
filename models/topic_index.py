from scipy.stats import entropy
from models.interfaces import TopicModel

class TextTopics:
    "Text with associated topic data"
    def __init__(self, id, index, weights):
        "Init with id, CorpusTopicsIndex, and weights - a mapping from integer to floats"
        self.id = id; self.weights = weights
        self.__createAttributes()

    def __createAttributes(self):
        "create derived attributes from topics"
        index = self.topic_ind
        mx = index[0]; mn = index[0]
        for i in index[1:] :
            if self.weights[i] > self.weights[mx] : mx = i
            if self.weights[i] < self.weights[mn] : mn = i

        self.max_topic, self.min_topic = mx, mn
        self.entropy = entropy([ self.weights[i] for i in index ])

class CorpusTopicsIndex :
    "Indexes texts in a corpus with a specific topic model data. Returns text ids, not texts"

    TOPIC_INDEX_FILE = 'corpusTopicIndex.pickle'

    def __init__(self, model, corpus, tokenizer):
        "init with indexes of topics"
        if not isinstance(model, TopicModel) : raise TypeError('model must be a TopicModel')
        self.corpus = corpus
        self.topic_ind = model.topic_indices()
        self.text_ids = []
        self.text_topics = {} # maping text id -> text topics
        for txt in corpus :
            self.text_ids.append(txt.id)
            topics = model.infer_topics(tokenizer(txt.text))
            self.text_topics[txt.id] = topics

    def getTextsForTopic(self, topic, topN = 10):
        'get topN (weight, document) pairs, for docs with largest weights of topic'
        doc_weigths = [ (i, self.text_topics[i][topic]) for i in self.text_ids ]
        doc_weigths.sort(key=lambda x: x[1])
        if topN > len(doc_weigths) : topN = len(doc_weigths)
        return [ dw for dw in doc_weigths[::-1][:topN] ]
