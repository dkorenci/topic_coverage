import ConfigParser
import os.path

moduleDir = os.path.dirname(__file__)
settingsFile = os.path.join(moduleDir, 'settings.ini')

config = ConfigParser.ConfigParser()
config.read(settingsFile)

models_folder = config.get('main', 'models_folder')
labeled_models_folder = config.get('main', 'labeled_models_folder')
snowball_stopwords = config.get('main', 'snowball_stopwords')
object_store = config.get('main', 'object_store')
themes_folder = config.get('main', 'themes_folder')
tfidfindex_folder = config.get('main', 'tfidfindex_folder')
labeling_folder = config.get('main', 'labeling_folder')
document_topics_cache = config.get('main', 'document_topics_cache')
default_corpus = config.get('main', 'default_corpus')

shuffle_documents = int(config.get('document_list', 'shuffle_documents'))
doctopic_threshold = float(config.get('document_list', 'doctopic_threshold'))

rsssucker_databases = [db for db in config.options('rsssucker_databases')
                       if config.get('rsssucker_databases', db) == '1']

def test():
    print config.sections()
    print config.get('main', 'var1')
