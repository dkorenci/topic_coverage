'''
Entry point for modelbuild runs, this is the script executed by bash runner script.
'''

from topic_coverage.resources.pytopia_context import topicCoverageContext

from pyutils.file_utils.location import FolderLocation as loc
from topic_coverage.modelbuild.modelbuild_docker_v1 import paramset1, paramset_lab, paramset_prod, paramsetTest1, \
    paramsetTest2, paramsetValid1, paramsetValid2, buildModels
from topic_coverage.modelbuild.modelbuild_numtopics import paramset_nmf_v0, paramset_prod_numt

def runBuild():
    import sys
    paramset = sys.argv[1]
    paramsetSplits = int(sys.argv[2])
    paramsetPart = int(sys.argv[3])
    bfolder = loc(sys.argv[4])()
    if paramset == 'paramset1': paramset = paramset1(True, paramsetSplits, 9871)
    elif paramset == 'paramset_lab_uspol':
        paramset = paramset_lab('uspol', True, paramsetSplits, 2, 83610)
    elif paramset == 'paramset_lab_pheno':
        paramset = paramset_lab('pheno', True, paramsetSplits, 2, 918468)
    elif paramset == 'paramset_prod':
        paramset = paramset_prod(True, paramsetSplits, numModels=10, rseed=8771203, rndmodel=False)
    elif paramset == 'paramsetTest1': paramset = paramsetTest1(True, paramsetSplits)
    elif paramset == 'paramsetTest2': paramset = paramsetTest2(True, paramsetSplits)
    elif paramset == 'paramsetValid1': paramset = paramsetValid1(True, paramsetSplits, 8772)
    elif paramset == 'paramsetValid2': paramset = paramsetValid2(True, paramsetSplits, 9174)
    elif paramset == 'nmfNumTopicsTest':
        paramset = paramset_nmf_v0(paramsetSplits, range(200, 301, 20),
                                   numModels=5, corpora=['pheno'], rseed=615)
    elif paramset == 'paramset_prod_numt':
        paramset = paramset_prod_numt(True, paramsetSplits, numModels=10, rseed=8976443)
    elif paramset == 'paramset_prod_numt_test':
        paramset = paramset_prod_numt(True, paramsetSplits,
                                      numModels=3, rseed=8764143, corpora=['uspol'],
                                      paramTopics=range(50, 101, 50),
                                      nonparamTopics=range(300, 301, 100)
                                      )
    else: raise Exception('unknown parameter set: %s' % paramset)
    print 'Paramset splits: param, real ', paramsetSplits, len(paramset)
    buildModels(paramset[paramsetPart], bfolder)

if __name__ == '__main__':
    with topicCoverageContext():
        runBuild()
