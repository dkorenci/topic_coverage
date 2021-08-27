from pytopia.utils.load import loadModels
from os import path

modelDir = path.join(path.dirname(__file__), 'models')

def loadTestModels():
    return loadModels(modelDir)
