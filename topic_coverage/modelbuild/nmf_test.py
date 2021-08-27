from topic_coverage.resources import pytopia_context
from pytopia.adapt.scikit_learn.nmf.adapter import SklearnNmfBuilder
from topic_coverage.settings import resource_folder

from os import path

from file_utils.location import FolderLocation as loc
modelfolder = loc(path.join(resource_folder, 'test_models'))

def buildNmf(corpus, dict, txt2tok, T):
    builder = SklearnNmfBuilder()
    nmf = builder(corpus=corpus, dictionary=dict, text2tokens=txt2tok, T=T)
    print nmf.id
    nmf.save(modelfolder(nmf.id))

def loadModels():
    from pytopia.resource.loadSave import loadResource
    models = []
    for f in modelfolder.subfolders():
        try: models.append(loadResource(f))
        except: pass
    return models

def printModels():
    for m in loadModels():
        print m

if __name__ == '__main__':
    #buildNmf('us_politics', 'us_politics_dict', 'RsssuckerTxt2Tokens', 50)
    printModels()



