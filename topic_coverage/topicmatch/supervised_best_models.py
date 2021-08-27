from topic_coverage.topicmatch.supervised_data import getLabelingContext, dataset
from topic_coverage.topicmatch.supervised_models import logistic, buildLoadSaveModel, switchOffMultiprocCache


def optModelV1(corpus='uspol', strict=True):
    ctx = getLabelingContext(corpus)
    with ctx:
        thresh = 0.75 if strict else 0.5
        dset = dataset(thresh, corpus, split=True)
        if corpus == 'uspol':
            mgrid = logistic # opt. solution for both strict and non-strict
        elif corpus == 'pheno':
            mgrid = logistic # opt. solution for both strict and non-strict
        mid = 'optModelV1_%s_%s'%(corpus, 'Strict' if strict else 'Nonstrict')
        res = buildLoadSaveModel(mid, dset, mgrid, 'allmetrics')
        print res
        return res

def modelIdLabel(label, corpus, strict, thresh, mgrid, features):
    mid = '%s_corpus[%s]_strict[%d]_thresh[%g]_model[%s]_features[%s]' \
          % (label, corpus, int(strict), thresh, mgrid.__name__, features)
    return mid

def optModelV2(corpus='uspol', strict=True, features='core1'):
    ctx = getLabelingContext(corpus)
    with ctx:
        thresh = 0.75 if strict else 0.5
        dset = dataset(thresh, corpus, split=True)
        if corpus == 'uspol':
            mgrid = logistic # opt. solution for both strict and non-strict
        elif corpus == 'pheno':
            mgrid = logistic # opt. solution for both strict and non-strict
        mid = modelIdLabel(optModelV2.__name__, corpus, strict, thresh, mgrid, features)
        res = buildLoadSaveModel(mid, dset, mgrid, features)
        switchOffMultiprocCache(res)
        return res

from topic_coverage.resources.pytopia_context import topicCoverageContext
if __name__ == '__main__':
    with topicCoverageContext():
        optModelV2()