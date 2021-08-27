from topic_coverage.modelbuild.modelbuild_docker_v1 import createParamset, psplit, shuffleAndSplit

def paramset_nmf_v0(numSplits, topics, corpora=['uspol', 'pheno'], numModels=10, rseed=981162):
    '''
    Params for building NMFs with a range of T's, numModels instances per T
    '''
    pset = []
    for corpus in corpora:
        pset.extend(createParamset(corpus, 'nmf', numModels, topics=topics, rseed=rseed))
    if numSplits: return shuffleAndSplit(pset, rseed, numSplits)
    else: return pset


def paramset_prod_numt(split, numSplits, corpora=['uspol', 'pheno'], numModels=10,
                       rseed=None, rndmodel=False, paramTopics=range(20, 501, 20),
                       nonparamTopics=range(100, 501, 100), ):
    '''
    Paramset for generating production models coverage ~ num.topics experiment.
    '''
    pset = []
    print paramTopics, len(paramTopics)
    print nonparamTopics, len(nonparamTopics)
    for corpus in corpora:
        for model in ['nmf', 'lda', 'lda-asym']:
            pset.extend(createParamset(corpus, model, numModels, topics=paramTopics,
                                       rseed=rseed, rndmodel=rndmodel))
        for model in ['pyp']:
            pset.extend(createParamset(corpus, model, numModels, topics=nonparamTopics,
                                       rseed=rseed, rndmodel=rndmodel))
    if split: return shuffleAndSplit(pset, rseed, numSplits, stratype=['lda-asym', 'pyp'])
    else: return pset

def paramset_prod_numt_sanitycheck(set='prod'):
    ''' Print smaller representative modelset for checking '''
    if set == 'prod':
        psplit(paramset_prod_numt(True, 5, numModels=4, rseed=121315,
                              paramTopics=range(20, 101, 20), nonparamTopics=range(50, 101, 50)))
    else:
        psplit(paramset_prod_numt(True, 3,
                           numModels=3, rseed=8764143, corpora=['uspol'],
                           paramTopics=range(50, 101, 50),
                           nonparamTopics=range(300, 301, 100)
                           ))

if __name__ == '__main__':
    #psplit(paramset_nmf_v0(4, range(20, 201, 20), numModels=2))
    #psplit(paramset_prod_numt(True, 1))
    paramset_prod_numt_sanitycheck('test')