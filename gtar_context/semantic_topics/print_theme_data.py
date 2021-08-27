from pytopia.context.ContextResolver import resolve

from gtar_context.semantic_topics.construct_model import \
    parseAllThemes, thisfolder, MODEL_ID, theme2topic
from gtar_context.compose_context import gtarContext
from gtar_context.builders_context import buildersContext

def newline2space(s):
    return u' '.join(unicode(s).split()).strip()

def printThemeData(theme, index):
    # print theme data: ref.table label, model topic labels and descriptions (load orig. models)
    print 'THEME %d; label: %s; dominant: %s' % (index, theme.label, theme.dominantData)
    for i, t in enumerate(theme.topics):
        model, topic = t.split('.')
        model = model.replace('~', '')
        model, topic = resolve(model), int(topic)
        print '  topic %d: %s, %s' % (i, t, model.topic2string(topic, topw=20))
        desc = model.description
        tlabel, tdesc = desc.topic[topic].label, desc.topic[topic].description
        tdesc = newline2space(str(tdesc))
        print '    label: %s ; description: %s' % (tlabel, tdesc)

def sampleAndPrintThemes(modelId=MODEL_ID, subsample=30, rseed=901442):
    '''
    Create three samples for labeling themes: one of size subsample (for join annotation/calibration),
    and another two, each containing half of the remaining topics (for annotation)
    '''
    from random import sample, seed, shuffle
    themes = parseAllThemes()
    def ids(ethemes): return [i for i, _ in ethemes]
    allthemes = [(i, th) for i, th in enumerate(themes)]
    printThemes(allthemes, modelId, title='ALLTHEMES')
    return
    seed(rseed)
    calibsample = sample(allthemes, subsample); calibIds = set(ids(calibsample))
    noncalibThemes = [(i, th) for i, th in allthemes if i not in calibIds]
    shuffle(noncalibThemes); N = len(noncalibThemes)/2
    part1, part2 = noncalibThemes[:N], noncalibThemes[N:]
    print 'lengths: calib %d, part1 %d, part2 %d' % (len(calibsample), len(part1), len(part2))
    printThemes(calibsample, modelId, title='CALIBRATION')
    printThemes(part1, modelId, title='PART1')
    printThemes(part2, modelId, title='PART2')
    assert len(ids(calibsample)+ids(part1)+ids(part2)) == len(allthemes)
    assert set(ids(calibsample)+ids(part1)+ids(part2)) == set(ids(allthemes))

def printThemes(themes, modelId=MODEL_ID, properties=True, title=None):
    from pytopia.resource.loadSave import loadResource
    from pytopia.resource.corpus_topics.CorpusTopicIndex import CorpusTopicIndexBuilder
    refmodel = loadResource(thisfolder(modelId))
    corpus = resolve(refmodel.corpus)
    if title: print '********************** %s **********************' % title
    for i, th in themes:
        printThemeData(th, i)
        # print refmodel topic-words and topic-docs
        if properties: print '[]'
        print 'WORDS: %s' % refmodel.topic2string(i, topw=20)
        builder = CorpusTopicIndexBuilder()  # resolve('corpus_topic_index_builder')
        cti = builder(corpus=refmodel.corpus, model=modelId)
        ids = [id for id, w in cti.topicTexts(i, top=100) if w != 0]
        titles = [txto.title if hasattr(txto, 'title') else txto.text[:100]
                    for txto in corpus.getTexts(ids)]
        titles = [newline2space(t) for t in titles if newline2space(t)]
        print 'DOCUMENTS: '
        print '\n'.join(titles)
        print

if __name__ == '__main__':
    with gtarContext():
        with buildersContext():
            sampleAndPrintThemes()