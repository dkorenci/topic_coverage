def assertModelsEqual(m1, m2):
    assert m1.id == m2.id
    assert sorted(m1.topicIds()) == sorted(m2.topicIds())
    for ti in m1.topicIds():
        t1, t2 = m1.topicVector(ti), m2.topicVector(ti)
        assert normalizeVector(t1) == normalizeVector(t2)

def normalizeVector(v):
    '''Make comparable representation from iterable of numbers'''
    return [ (i, val) for i, val in enumerate(v) ]

def createSaveLoadCompare(builder, params, savedir):
    '''
    Basic testing of pytopia resources: create a resource, save and load it,
     then assert equality of the created and loaded resource.
    Also tests that creation/loading/saving do not break.
    :param params: build params, either a list or a single param
    '''
    from os import path
    from pytopia.resource.loadSave import saveResource, loadResource
    if not isinstance(params, list): params = [params]
    for p in params:
        res = builder(**p)
        assert res is not None
        folder = path.join(savedir, res.id)
        saveResource(res, folder)
        lres = loadResource(folder)
        assert lres is not None
        assert res.id == lres.id
        if hasattr(res, '__eq__'):
            assert res == lres

def flattenParams(params={}):
    '''
    :param params: map key -> value or a list of values
    :return: list of maps each representing one of the possible combinations
             where each key has only one possible value assigned to it
    '''
    keys = params.keys(); NK = len(keys)
    res = []; cm = {}
    def fillMapRecursive(ki):
        if ki == NK:
            res.append(cm.copy())
            return
        key = keys[ki]
        val = params[key]
        if not isinstance(val, list):
            cm[key] = val
            fillMapRecursive(ki+1)
        else:
            for v in val:
                cm[key] = v
                fillMapRecursive(ki + 1)
    fillMapRecursive(0)
    return res

def flattenParamList(plist):
    return [ fp for p in plist for fp in flattenParams(p) ]

def testFlatten():
    fl = flattenParams({'k1':['v1.1','v1.2'], 'k2':'v2', 'k3':['v3.1','v3.2','v3.3']})
    for p in fl: print p

def joinParams(pl1, pl2):
    '''
    :param p1, pl2: list of maps
    :return: list of maps, each being 'sum' of m1 \in p1, m2 \in p2,
            for all combinations
    '''
    res = []
    {}.update()
    for p1 in pl1:
        for p2 in pl2:
            np = p1.copy(); np.update(p2)
            res.append(np)
    return res

def testJoin():
    pl1 = [ {'a':1}, {'a':2} ]
    pl2 = [ {'b':1}, {'b':2} ]
    for p in joinParams(pl1, pl2): print p

def compareCorpora(corpus1, corpus2, testProperties=False, txt2tok=None, id2str=False,
                                     prop2str=False):
    '''
    Compare two corpora for equality, ie if they contain same Text objects with same ids
    :param testProperties: if True, test all text properties, else compare only .text
    :param txt2tok: if not None, a pair of text2tokens processors,
            in which case compare tokenized texts
    :param id2str: if True, convert all ids to string before comparison
    :param prop2str: if True, convert all properties to string before comparison
    :return:
    '''
    from pytopia.context.ContextResolver import resolve
    # test equality of the id sets
    idtrans = lambda i: unicode(i) if id2str else i
    id2txt = lambda c: { idtrans(txto.id) : txto for txto in c }
    corpus1, corpus2 = resolve(corpus1, corpus2)
    id2txt1 = id2txt(corpus1); ids1 = set(i for i in id2txt1)
    print ids1
    id2txt2 = id2txt(corpus2); ids2 = set(i for i in id2txt2)
    print ids2
    assert ids1 == ids2
    # resolve tokenizers
    if txt2tok is not None:
        tok1, tok2 = txt2tok
        tok1, tok2 = resolve(tok1), resolve(tok2)
    else:
        tok1, tok2 = None, None
    # test text equality
    def tokenize(txto, txt2tok):
        if txt2tok is None: return txto.text
        return ' '.join(tok for tok in txt2tok(txto.text))
    prtrans = lambda p: unicode(p).strip() if prop2str else p
    def propMap(txto): # create name->val map of Text properties
        return { name: prtrans(val) for name, val in txto }
    def propertiesEqual(txto1, txto2):
        pmap1, pmap2 = propMap(txto1), propMap(txto2)
        assert set(n for n in pmap1) == set(n for n in pmap2)
        for n in pmap1:
            if pmap1[n] != pmap2[n]:
                print pmap1
                print pmap2
                print pmap1[n]
                print pmap2[n]
            assert pmap1[n] == pmap2[n]
    for i in ids1:
        txto1, txto2 = id2txt1[i], id2txt2[i]
        assert tokenize(txto1, tok1) == tokenize(txto2, tok2) # text comparison
        if testProperties:
            propertiesEqual(txto1, txto2)
        #todo property test, with optional 'normalization' by converting to string
