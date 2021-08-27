import corpus.textstream.textfiles as txtfiles
from corpus.rsssucker import RsssuckerCorpus

def writeRssSuckerDBToFolder(database, folder):
    dbstream = RsssuckerCorpus(database)
    txtfiles.writeTextsToFolder(dbstream, folder)
        
    