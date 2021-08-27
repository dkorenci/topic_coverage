from topic_coverage.resources import pytopia_context
from topic_coverage.modelbuild.modelbuild_iter1 import loadModels, \
        uspolModelFolders, phenoModelFolders, allModelFolders, addModelsToGlobalContext
from topic_coverage.topicmatch.topicplots import valueDist
from topic_coverage.topicmatch.distance_sampling import *
from pytopia.measure.topic_distance import cosine
from topic_coverage.settings import resource_folder
from pyutils.file_utils.location import FolderLocation as loc

import numpy as np

def generateIter0LabelingSet():
    from topic_coverage.topicmatch.pair_labeling import \
            topicLabelText, createLabelingFiles, parseLabelingFolder, parseLabelingFile
    print 'adding models to context ...'
    addModelsToGlobalContext()
    print 'done.'
    lfolder = loc(resource_folder)('topicmatch', 'labeling_iter0')
    createLabelingFiles(lfolder, 'iter0Uspol', iter0UspolPairs,
                        intervals(0, 1, 10), 100, docs=True, filesize=50, rndseed=988715)

#def iter0PhenoLabelingModels():

def iter0PhenoModels(context=False):
    from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
    cacheFolder = '/datafast/topic_coverage/docker_modelbuild/djurdja_build1/'
    top = 'T[100]';
    corpus = 'pheno_corpus1'
    msets = [[top, corpus, 'lda-asym'], ['Nmf', top, corpus]]
    models = [m for f in msets for m in
              ResourceBuilderCache.loadResources(cacheFolder, filter=f, asContext=False)]
    if context:
        from pytopia.context.Context import Context
        ctx = Context('iter0PhenoModelsContext')
        for m in models: ctx.add(m)
        return ctx
    else: return models

def generateIter0PhenoLabelingSet(action='sample_distances'):
    if action == 'create_pairs':
        topics = [t for m in iter0PhenoModels() for t in m]
        createDistances(topics, cosine, 63156, 'phenoTopicsIter0', verbose=True)
    elif action == 'create_labeling':
        from topic_coverage.topicmatch.pair_labeling import createLabelingFiles
        #addModelsToGlobalContext()
        lfolder = loc(resource_folder)('topicmatch', 'labeling_iter0_pheno')
        with iter0PhenoModels(True):
            createLabelingFiles(lfolder, 'phenoTopicsIter0', 'topicDist_[topics=phenoTopicsIter0]_[dist=cosine]_[seed=63156]',
                            intervals(0, 1, 10), 100, docs=True, filesize=50, rndseed=988715)

if __name__ == '__main__':
    #generateIter0LabelingSet()
    #generateIter0PhenoLabelingSet('create_pairs')
    generateIter0PhenoLabelingSet('create_labeling')