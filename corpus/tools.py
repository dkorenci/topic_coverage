from utils.utils import normalize_path
from experiments.labeling.tools import formatText
from preprocessing.text2tokens import RsssuckerTxt2Tokens

import codecs, random

def rsssuckerCorpusToFolder(corpus, folder):
    folder = normalize_path(folder)
    txt2tok = RsssuckerTxt2Tokens()
    for txto in corpus:
        filename = folder + str(txto.id) + '.txt'
        f = codecs.open(filename, 'w', 'utf-8')
        f.write('ID: %d\n' % txto.id)
        f.write('TITLE: %s\n' % txto.title)
        f.write('URL: %s\n' % txto.url)
        f.write('DATE SAVED: %s\n' % txto.date.strftime('%d.%m.%Y %H:%M:%S'))
        text = formatText(' '.join(txt2tok(txto.text)))
        f.write('STEMMED WORDS:\n%s\n' % text)

def rsssuckerFullCorpusToFolder(corpus, folder, sample = None):
    folder = normalize_path(folder)
    txt2tok = RsssuckerTxt2Tokens()
    if sample is None: texts = [txto for txto in corpus]
    else:
        all_ids = [ id for id in corpus.getIds() ]
        random.seed(34567)
        random.shuffle(all_ids)
        texts = [txto for id, txto in corpus.getTexts(all_ids[:sample])]
    for txto in texts:
        filename = folder + str(txto.id) + '.txt'
        f = codecs.open(filename, 'w', 'utf-8')
        f.write('%s\n' % txto.title)
        text = formatText(txto.text)
        f.write('%s\n' % text)