'''
Entry point for modelbuild runs, this is the script executed by bash runner script.
'''

from topic_coverage.resources.pytopia_context import topicCoverageContext
from topic_coverage.experiments.modelparams.calc_coverage import calcCoverage

from os.path import join

def runCalcPlot():
    import sys
    print "PARAMS"
    print sys.argv
    print
    modelfold = sys.argv[1]
    cacheFolder = sys.argv[2]
    corpus = sys.argv[3] # uspol, pheno
    covfunc = sys.argv[4] # sup.strict, ctc
    modeltyp = sys.argv[5] # ['lda', 'alda', 'nmf', 'pyp']
    supCovCache = join(cacheFolder, 'sup_model_coverage')
    ctcCovCache = join(cacheFolder, 'ctc_model_coverage')
    calcCoverage(folders=[modelfold], corpus=corpus, numT=None, modelFamilies=[modeltyp],
                 covTyp=covfunc, supCovCache=supCovCache, ctcCovCache=ctcCovCache)

if __name__ == '__main__':
    with topicCoverageContext():
        runCalcPlot()
