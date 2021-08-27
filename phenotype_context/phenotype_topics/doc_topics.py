'''
(re)Construction of topic-document distribution from original resources.
'''

from pytopia.context.ContextResolver import resolve

from phenotype_context.settings import origClustersFolder, corpusLabels, \
    topicBacteriaFolder, pheno2clusters

from pyutils.file_utils.location import FolderLocation
import codecs, re, decimal, os, numpy as np

def corpNameNorm(cname):
    ''' Normalize corpus name, corpora are designated with varying strings depending on data file. '''
    if cname == 'PubMed abstracts:': cname = 'pubmedSearch'
    cname = cname.strip().replace(':', '').replace(' ', '').lower()
    return cname

def corpusIdIndex(corpus):
    import random
    corpus = resolve(corpus)
    txtIds = [ txto.id for txto in corpus ]
    #random.shuffle(txtIds)
    #for id in txtIds[:100]: print id
    def parseTid(tid):
        '''
        Parse text id into corpus+bacteria data sufficient for retrieving the id.
        :return: corpus name, taxa id
        '''
        #assert '(' in tid and ')' in tid
        #print tid
        assert re.match(r'[^\(\)]+\([0-9]+\)', tid)
        corpBact = tid.split('__')[0]
        corp = corpNameNorm(corpBact.split('_')[0]) # take corpus name and normalize
        taxid = int(tid.split('(')[1][:-1] ) # take number between paranthesis
        #print corp, taxid
        return corp, taxid
    cidx = {}
    for tid in txtIds:
        idKey = parseTid(tid)
        if idKey not in cidx: cidx[idKey] = []
        cidx[idKey].append(tid)
    # for id in cidx:
    #     if len(cidx[id]) > 1:
    #         print id
    #         for i in cidx[id]: print '\t', i
    #         print
    return cidx

def createDocTopicMatrix(phenoParse, corpusId):
    corpus = resolve(corpusId)
    cidx = corpusIdIndex(corpus)
    cib = resolve('corpus_index_builder')
    ci = cib(corpus)
    N, M = len(phenoParse), len(ci)
    print 'LENGTHS', len(corpus), len(ci)
    topicDocMatrix = np.zeros((N, M), np.float32)
    pheno2clustId = parsePheno2Cluster(pheno2clusters)
    def parseClusterId(cid):
        sp = cid.split('_')
        return sp[0], sp[1], sp[2]+'_'+sp[3]
    for phId, phParse in phenoParse.iteritems():
        handle = '(' + ','.join(word for word, _ in phParse['CENTROID'][:5]) + ')'
        model, seed, cluster = parseClusterId(pheno2clustId[handle])
        corpus2bact = parseClusterFactors(topicBacteriaFolder, model, seed, cluster)
        numCorp = 0
        for corpName in phParse:
            if corpName == 'CENTROID': continue
            cname = corpNameNorm(corpName)
            if cname not in corpus2bact: continue
            # construct vector of corpus-level textId weights for the phenotype sub-corpus
            cvec = np.zeros(M, np.float32); bactResolved = 0
            for bact, taxid, w in corpus2bact[cname]:
                if (cname, taxid) in cidx:
                    bactResolved += 1
                    ids = cidx[(cname, taxid)]
                    for tid in ids: cvec[ci.id2index(tid)] = w
                else:
                    print 'no id for: ', (cname, taxid)
            if bactResolved > 0:
                topicDocMatrix[phId] += cvec
                numCorp += 1
        if numCorp > 0: topicDocMatrix[phId] /= numCorp # average sub-corpus vectors
        else:
            print 'NO CORPORA RESOLVED FOR', phId, handle
    return topicDocMatrix.T

def parsePheno2Cluster(f):
    ph2clust = {}
    for i, l in enumerate(codecs.open(f, 'r', 'utf-8').readlines()):
        sp = l.split()
        clustId, handle = sp[0], sp[1]
        ph2clust[handle] = clustId
    return ph2clust

corporaNames = set()
def addCorpusName(cn):
    global corporaNames
    corporaNames.add(corpNameNorm(cn))

def parseBacteriaData(fpath):
    '''Parse file with factor-bacteria weights, return list of
     (bacteria name, taxa id, weight)'''
    bacts = []
    for i, l in enumerate(codecs.open(fpath, 'r', 'utf-8').readlines()):
        if i == 0: continue # skip header
        l = l.strip()
        sp = l.split()
        bact, taxId, weight = sp[0], int(sp[1]) ,float(sp[2])
        bacts.append((bact, taxId, weight))
    return bacts

def parseClusterFactors(folder, model, seed, cluster, verbose=False):
    '''
    :return: corpusName -> bacteria weight list
    '''
    import glob
    from os import path
    def parseCorpusName(fname): # cluster_28_hamap.txt -> hamap
        cn = fname.split('_')[2]
        return cn.split('.')[0]
    folder = path.join(folder, model, seed)
    corp2bact = {}
    for f in glob.glob1(folder, cluster+'_*'):
        corp = corpNameNorm(parseCorpusName(f))
        if verbose: print f, parseCorpusName(f)
        bacts = parseBacteriaData(path.join(folder, f))
        corp2bact[corp] = bacts
    return corp2bact

if __name__ == '__main__':
    createDocTopicMatrix()
