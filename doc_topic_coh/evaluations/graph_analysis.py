from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from pytopia.measure.topic_distance import cosine
from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit, devTestSplit2
from doc_topic_coh.evaluations.iteration2.best import cacheFolder

import networkx as nx

def buildGraph(edges, weighted=False):
    '''
    Build networkx graph from a list of edges - (vert1, vert2, weight) tuples
    '''
    g = nx.Graph()
    nodes = set(e[0] for e in edges).union(e[1] for e in edges)
    for n in nodes: g.add_node(n)
    for e in edges:
        if weighted: g.add_edge(e[0], e[1], weight=e[2])
        else: g.add_edge(e[0], e[1])
    return g

def analyzeGraphs():
    bestGraph1 = {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'communicability',
                 'vectors': 'probability', 'threshold': 50, 'weightFilter': [0, 0.95], 'type': 'graph'}
    bestGraph2 = {'distance': cosine, 'weighted': False, 'center': 'mean', 'algorithm': 'closeness',
                  'vectors': 'probability', 'threshold': 50, 'weightFilter': [0, 0.95], 'type': 'graph'}
    graph1 = {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'communicability',
                 'vectors': 'probability', 'threshold': 100, 'weightFilter': [0, 0.95], 'type': 'graph'}
    cohParams = graph1
    #cohParams['cache'] = cacheFolder
    coh = DocCoherenceScorer(**cohParams)()
    dev, test = devTestSplit2()
    tset = dev
    for t, tl in tset:
        coh(t)

def analyzeCentrality(graph, measure, params={}):
    from numpy import average as avg
    nodeCentr = measure(graph, **params)
    print str(measure.__name__)
    print 'average centrality: %.4f' % avg(nodeCentr.values())
    for node, cv in nodeCentr.iteritems():
        print node, '%.4f' % cv

from networkx.algorithms.centrality import closeness_centrality as closec
from networkx.algorithms.centrality import communicability_centrality as commc
def analyze():
    from networkx import minimum_spanning_tree as mst
    g1 = [(1,2), (3,4), (5,6)]
    g1 = buildGraph(g1)
    g2 = [(1,2,1), (2,3,1), (3,4,1), (1,3,2), (1,4,2)]
    g2 = buildGraph(g2, True)
    #analyzeCentrality(g1, closec)
    #analyzeCentrality(g1, commc)
    mt = mst(g2)
    for n1, n2 in mt.edges():
        print g2.get_edge_data(n1,n2)['weight']

if __name__ == '__main__':
    analyze()
    #analyzeGraphs()