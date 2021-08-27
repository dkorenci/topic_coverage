from pytopia.context.Context import Context
from topic_coverage.modelbuild.modelbuild_docker_v1 import modelset, msetFilter

from topic_coverage.settings import topic_models_folder

def modelset1Families(corpus='uspol', numModels=5, modelsFolder=topic_models_folder,
                      families='all', numT = [50, 100, 200]):
    if corpus == 'uspol': corpus = 'us_politics_textperline'
    elif corpus == 'pheno': corpus = 'pheno_corpus1'
    if families == 'all': families = ['lda', 'alda', 'nmf', 'pyp']
    filters = []
    for typ in ['lda', 'alda', 'nmf']:
        if typ in families: filters += [[typ, nt, corpus] for nt in numT]
    if 'pyp' in families: filters += [['pyp', 300, corpus]]
    #print filters
    modelsets = []; labels = []; modelCtx = Context('loaded_models_context')
    for f in filters:
        mset = modelset(modelsFolder, msetFilter(*f), num=numModels)
        for m in mset: modelCtx.add(m)
        modelsets.append(mset)
        labels.append('%s[%d]_%s'%(f[0],f[1],f[2]))
    return modelsets, modelCtx, labels

def modelsetLoad(corpus='uspol', modelsFolder=topic_models_folder,
                      families='all', numT = [50, 100, 200], autoseg=False):
    if corpus == 'uspol': corpus = 'us_politics_textperline'
    elif corpus == 'pheno': corpus = 'pheno_corpus1'
    if families == 'all': families = ['lda', 'alda', 'nmf', 'pyp']
    filters = []
    for typ in ['lda', 'alda', 'nmf', 'pyp']:
        if typ in families:
            if numT: filters += [[typ, nt, corpus] for nt in numT]
            else: filters += [[typ, None, corpus]]
    #print filters
    modelsets = []; labels = []; modelCtx = Context('loaded_models_context')
    for f in filters:
        mset = modelset(modelsFolder, msetFilter(*f), num=None)
        for m in mset: modelCtx.add(m)
        if (numT == None and autoseg):
            byNumt = {m.numTopics():[] for m in mset}
            for m in mset: byNumt[m.numTopics()].append(m)
            for ms in byNumt.values():
                if (len(ms) > 1): modelsets.append(ms)
        else:
            modelsets.append(mset)
        labels.append('%s[%s]_%s'%(f[0],f[1],f[2]))
    return modelsets, modelCtx, labels

def printModelset(mset):
    for m in mset: print m.id