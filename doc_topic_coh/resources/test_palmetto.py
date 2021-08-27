from doc_topic_coh.resources import pytopia_context

def testPytopiaPalmetto():
    from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
    #dcs = DocCoherenceScorer('npmi', index='wiki_docs', windowSize=0, standard=False)
    dcss = [
        # DocCoherenceScorer('npmi', index='wiki_docs', windowSize=15, standard=False),
        # DocCoherenceScorer('npmi', index='wiki_paragraphs', windowSize=0, standard=False),
        # DocCoherenceScorer('c_v', index='wiki_paragraphs', windowSize=0, standard=False),
        # DocCoherenceScorer('c_a', index='wiki_docs', windowSize=5, standard=False),
        # DocCoherenceScorer('c_a', index='wiki_docs', windowSize=100, standard=False),
        DocCoherenceScorer('c_a', index='wiki_standard', standard=True),
        # DocCoherenceScorer('c_a', index='wiki_standard', windowSize=50, standard=True)
    ]
    cohs = [ dcs() for dcs in dcss ]
    for coh in cohs:
        print coh.id
        print coh(('uspolM0', 4))
        print coh(('uspolM10', 64))

if __name__ == '__main__':
    testPytopiaPalmetto()