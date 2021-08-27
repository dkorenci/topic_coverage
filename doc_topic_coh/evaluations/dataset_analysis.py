from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit, topicLabelStats
from doc_topic_coh.dataset.topic_labels import labelAllTopics, labelingStandard
from doc_topic_coh.dataset.croelect_dataset import labeledTopics

# theme, theme_noise, theme_mix, theme_mix_noise, noise
dev, test = iter0DevTestSplit()
allTopic = labelAllTopics(labelingStandard)

def croelectStats():
    alltopics = labeledTopics(['croelect_model1', 'croelect_model2', 'croelect_model3'])
    topicLabelStats(alltopics)

def uspolStats():
    topicLabelStats(labelAllTopics(labelingStandard)); print
    topicLabelStats(dev); print
    topicLabelStats(test)

if __name__ == '__main__':
    croelectStats()