from resources.builder import ResourceBuilder
from corpus.factory import CorpusFactory
from pymedialab_settings.settings import object_store, models_folder
from preprocessing.text2tokens import RsssuckerTxt2Tokens

def resourceBuilder():
    return ResourceBuilder(corpusFactory=CorpusFactory, objectStore=object_store,
                modelsFolder=models_folder, text2tokens=RsssuckerTxt2Tokens())

def customResourceBuilder(models_folder, object_store):
    return ResourceBuilder(corpusFactory=CorpusFactory, objectStore=object_store,
                modelsFolder=models_folder, text2tokens=RsssuckerTxt2Tokens())