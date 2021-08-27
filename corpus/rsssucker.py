from text import Text
from corpus import Corpus
from preprocessing.tokenizers import prefilter_tokenizer

from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import random

Base = declarative_base()
class Article(Base):
    "ORM class mapping to feedarticle table"
    __tablename__ = 'feedarticle'
    id = Column(Integer, primary_key=True)
    text = Column(String)
    feedtitle = Column(String)
    datesaved = Column(DateTime)

class RsssuckerCorpus(Corpus):
    'corpus of all text documents in a specified rsssucker database'

    def __init__(self, dbName, uname='rsssucker', pword='rsssucker', id = None):
        self._dbname = dbName; self._uname = uname ; self._password = pword
        self.id = id
        self._dbConnected = False

    def _initDB(self):
        if self._dbConnected : return
        conn_str = "postgresql+psycopg2://%s:%s@localhost/%s"%\
                   (self._uname, self._password, self._dbname)
        self._engine = create_engine(conn_str)
        self._session_maker = sessionmaker(bind=self._engine)
        # todo thread safety of one global session?
        self._session = self._session_maker()
        self._dbConnected = True

    def __getstate__(self):
        return self._dbname, self._uname, self._password, self.id

    def __setstate__(self, state):
        self._dbname, self._uname, self._password, self.id = state
        self._dbConnected = False

    def corpusId(self):
        if self.id is not None: return self.id
        else: return self._dbname

    def getIds(self):
        self._initDB()
        " read all ids from the database "
        session = self._session_maker()
        for id in session.query("id").execution_options(stream_result=True).\
                    from_statement(text("SELECT id FROM feedarticle")).all() :
            yield id[0]
        session.close()

    def getTexts(self, id_list):
        self._initDB()
        "yield (id, text) pairs for text with specified ids"
        # create list of ids for sql query
        if len(id_list) == 0 : return
        id_str = "(";
        for id in id_list : id_str += (str(id)+',')
        id_str = id_str[:-1]; id_str += ')'
        query = "SELECT id, text, feedtitle FROM feedarticle WHERE id IN %s" % id_str
        session = self._session #self._session_maker()
        for id, txt, title in session.query("id","text","feedtitle").\
                    execution_options(stream_result=True).from_statement(text(query)).all() :
            txt = Text(id, txt); txt.title = title
            yield (id, txt)
        #session.close()

    def getFeeds(self, txto):
        '''
        Fetch list of urls of feeds containing this text.
        "Save" to txto.feedUrls property
        '''
        template = '''
        SELECT DISTINCT feed.url FROM feed
        LEFT JOIN feedarticle_feed ff ON ff.feeds_id = feed.id
        WHERE articles_id = %d
        '''
        self._initDB()
        session = self._session_maker()
        query = template % txto.id
        result = []
        for url in session.query('url').execution_options().from_statement(text(query)).all():
            result.append(url[0])
        session.close()
        txto.feedUrls = result

    def getOutlets(self, txto):
        '''
        Fetch list of names of outlets containing this text.
        "Save" to txto.outlets property
        '''
        template = '''
        select distinct o.name from outlet o
        left join feed f on f.outlet_id = o.id
        left join feedarticle_feed ff on ff.feeds_id = f.id
        where ff.articles_id = %d
        '''
        self._initDB()
        session = self._session_maker()
        query = template % txto.id
        result = []
        for name in session.query('name').execution_options().from_statement(text(query)).all():
            result.append(name[0])
        session.close()
        txto.outlets = result

    def getSample(self, size = 100, seed = 12345):
        random.seed(seed)
        ids = [i for i in self.getIds()]; random.shuffle(ids)
        id_sample = ids[:size]
        for idtxt in self.getTexts(id_sample):
            yield idtxt[1]

    def __iter__(self):
        " opens a session to the database and iterates over all article texts "
        self._initDB()
        session = self._session_maker()
        cnt = 0
        #for row in session.query(Article.text):
        for id, txt, title in session.query("id","text","feedtitle").\
                    execution_options(stream_result=True).\
                    from_statement(text("SELECT id, text, feedtitle FROM feedarticle")).all() :
            txt = Text(id, txt); txt.title = title
            yield txt
        session.close()

    def __len__(self):
        if not hasattr(self, 'length'):
            # cache the corpus length
            self.length = sum(1 for _ in self.getIds())
        return self.length

class FeedsetCorpus(RsssuckerCorpus):
    '''
    corpus of the text documents in the rsssucker database which belong
    to a specified set of seed, plus date filter
    '''
    # db query strings, parametrized by the set of fetched columns,
    # set of feed urls and aditional article conditions
    # todo: refactor, pull out only neccessary query parameters

    # this is hardcoded default, for old pickled corpuses with version
    # where this attribute was not set in the constructor
    query_template = '''
    SELECT %s FROM feedarticle WHERE
        (datepublished > '2015-01-26 00:00:00' OR
        (datepublished IS NULL AND datesaved > '2015-01-26 00:00:00'))
        AND
        (id IN (SELECT DISTINCT art.id AS feed_id FROM feedarticle AS art
        JOIN (
        SELECT DISTINCT articles_id, feeds_id FROM feedarticle_feed WHERE feeds_id IN
            ( SELECT id FROM feed WHERE url IN %s )
        ) AS t ON art.id = t.articles_id))
        %s
    '''
    gtar_query_template = '''
    SELECT %s FROM feedarticle WHERE
        (datepublished > '2015-01-26 00:00:00' OR
        (datepublished IS NULL AND datesaved > '2015-01-26 00:00:00'))
        AND
        (id IN (SELECT DISTINCT art.id AS feed_id FROM feedarticle AS art
        JOIN (
        SELECT DISTINCT articles_id, feeds_id FROM feedarticle_feed WHERE feeds_id IN
            ( SELECT id FROM feed WHERE url IN %s )
        ) AS t ON art.id = t.articles_id))
        %s
    '''
    croelect_query_template = '''
    SELECT %s FROM feedarticle WHERE
        (datepublished > '2015-09-30 00:00:00' OR
        (datepublished IS NULL AND datesaved > '2015-09-30 00:00:00'))
        AND
        (datepublished < '2015-12-28 23:59:59' OR
        (datepublished IS NULL AND datesaved < '2015-12-28 23:59:59'))
        AND
        (id IN (SELECT DISTINCT art.id AS feed_id FROM feedarticle AS art
        JOIN (
        SELECT DISTINCT articles_id, feeds_id FROM feedarticle_feed WHERE feeds_id IN
            ( SELECT id FROM feed WHERE url IN %s )
        ) AS t ON art.id = t.articles_id))
        %s
    '''
    all_articles_template = '''
    SELECT %s FROM feedarticle WHERE
        (id IN (SELECT DISTINCT art.id AS feed_id FROM feedarticle AS art
        JOIN (
        SELECT DISTINCT articles_id, feeds_id FROM feedarticle_feed WHERE feeds_id IN
            ( SELECT id FROM feed WHERE url IN %s )
        ) AS t ON art.id = t.articles_id))
        %s
    '''
    #todo add date and random seed to params
    def __init__(self, dbName, feedlist, template='gtar', uname='rsssucker', pword='rsssucker', id = None):
        RsssuckerCorpus.__init__(self, dbName, uname, pword)
        if template == 'gtar':
            self.query_template = self.gtar_query_template
        elif template == 'croelect': self.query_template = self.croelect_query_template
        elif template == 'all': self.query_template = self.all_articles_template
        else: self.query_template = template
        self.feedlist = feedlist; self.id = id

    #todo solve getstate and setstate more elegantly via superclass methods
    #TODO add query_template to get and set state, and rebuild corpus indexes
    #  as is, this could cause errors when saving/loading corpus indexes
    def __getstate__(self):
        return self._dbname, self._uname, self._password, self.feedlist, self.id

    def __setstate__(self, state):
        self._dbname, self._uname, self._password, self.feedlist, self.id = state
        self._dbConnected = False

    def corpusId(self):
        if self.id is not None: return self.id
        else:
            return RsssuckerCorpus.corpusId(self) + '_feedset_' + self.feedlist.id

    def getFeedSet(self):
        'get set of feed urls for sql queries'
        return '(' + ','.join(["'%s'"%url for url in self.feedlist.urls]) + ')'

    def getQueryTemplate(self, randomOrder=False):
        if randomOrder: return \
            'SELECT setseed(0.1);\n'+self.query_template+'\n ORDER BY random()'
        else: return self.query_template

    def getIds(self):
        'read all ids from the filtered database'
        #RsssuckerCorpus._initDB(self)
        self._initDB()
        session = self._session_maker()
        query = self.getQueryTemplate() % ('id' , self.getFeedSet(), '')
        for id in session.query('id').execution_options(stream_result=True).\
                    from_statement(text(query)).all() :
            yield id[0]
        session.close()

    def getTexts(self, id_list):
        self._initDB()
        'yield (id, text) pairs for text with specified ids'
        # create list of ids for sql query
        if len(id_list) == 0 : return
        id_str = '('+','.join([str(i) for i in id_list])+')'
        query = self.getQueryTemplate() \
                % ('id, text, feedtitle, datesaved, datepublished, url',
                   self.getFeedSet(), 'AND id IN %s' % id_str)
        session = self._session #self._session_maker()
        for id, txt, title, datesav, datepub, url in \
                session.query('id','text','feedtitle','datesaved', 'datepublished', 'url').\
                execution_options(stream_result=True).from_statement(text(query)).all() :
            txt = Text(id, txt); txt.title = title; txt.date = datesav; txt.url = url
            txt.datesaved = datesav; txt.datepublished = datepub
            yield (id, txt)
        session.close()

    def __iter__(self):
        " opens a session to the database and iterates over all article texts "
        self._initDB()
        session = self._session_maker()
        cnt = 0
        #for row in session.query(Article.text):
        query = self.getQueryTemplate(True) \
                % ('id, text, feedtitle, datesaved, datepublished, url', self.getFeedSet(), '')
        #print query
        for id, txt, title, datesav, datepub, url in session.query('id','text','feedtitle',
                                                       'datesaved', 'datepublished', 'url').\
                    execution_options(stream_result=True).from_statement(text(query)).all() :
            txt = Text(id, txt); txt.title = title; txt.date = datesav; txt.url = url
            txt.datesaved = datesav; txt.datepublished = datepub
            yield txt
        session.close()

class IdsetCorpus(RsssuckerCorpus):
    '''
    corpus of the text documents in the rsssucker database
    defined by the set of ids.
    '''
    # db query string, parametrized by the set of fetched columns,
    # set of feed urls and aditional article conditions
    query_template = '''
    SELECT %s FROM feedarticle WHERE id in %s
    '''
    #todo add date and random seed to params
    def __init__(self, dbName, idSet, uname='rsssucker', pword='rsssucker', id = None):
        RsssuckerCorpus.__init__(self, dbName, uname, pword)
        self.idSet = idSet; self.id = id

    #todo solve getstate and setstate more elegantly via superclass methods
    def __getstate__(self):
        return self._dbname, self._uname, self._password, self.idSet, self.id

    def __setstate__(self, state):
        self._dbname, self._uname, self._password, self.idSet, self.id = state
        self._dbConnected = False

    def corpusId(self):
        if self.id is not None: return self.id
        else:
            return RsssuckerCorpus.corpusId(self) + '_idSet'

    def getIdSet(self):
        'get set of ids for sql queries'
        return '(' + ','.join(['%d' % id for id in self.idSet]) + ')'

    def getQueryTemplate(self, randomOrder=False):
        if randomOrder: return \
            'SELECT setseed(0.1);\n'+self.query_template+'\n ORDER BY random()'
        else: return self.query_template

    def getIds(self):
        'read all ids from the filtered database'
        #RsssuckerCorpus._initDB(self)
        self._initDB()
        session = self._session_maker()
        query = self.getQueryTemplate() % ('id' , self.getIdSet())
        for id in session.query('id').execution_options(stream_result=True).\
                    from_statement(text(query)).all() :
            yield id[0]
        session.close()

    def __iter__(self):
        " opens a session to the database and iterates over all article texts "
        self._initDB()
        session = self._session_maker()
        query = self.getQueryTemplate(True) \
                % ('id, text, feedtitle, datesaved, datepublished, url', self.getIdSet())
        #print query
        for id, txt, title, datesav, datepub, url in session.query('id','text','feedtitle',
                                                       'datesaved', 'datepublished', 'url').\
                    execution_options(stream_result=True).from_statement(text(query)).all() :
            txt = Text(id, txt); txt.title = title; txt.date = datesav; txt.url = url
            txt.datesaved = datesav; txt.datepublished = datepub
            yield txt
        session.close()

    def getTexts(self, id_list):
        '''
        yield (id, text) pairs for text with specified ids
        '''
        self._initDB()
        # create list of ids for sql query
        ids = set(i for i in id_list);
        ids.intersection_update(self.idSet)
        if len(ids) == 0 : return
        idStr = '('+','.join([str(i) for i in ids])+')'
        query = self.query_template % ('id, text, feedtitle, datesaved, datepublished, url', self.getIdSet())
        query = query + (' AND id IN %s' % idStr)
        session = self._session
        for id, txt, title, datesav, datepub, url in \
                session.query('id','text','feedtitle','datesaved', 'datepublished', 'url').\
                    execution_options(stream_result=True).from_statement(text(query)).all() :
            txt = Text(id, txt); txt.title = title; txt.date = datesav; txt.url = url
            txt.datesaved = datesav; txt.datepublished = datepub
            yield (id, txt)
        session.close()

class Feedlist():

    def __init__(self, file):
        lines = open(file).readlines()
        self.id = lines[0].strip() # set id is on the first line
        self.urls = set(l.strip() for l in lines[1:] if l.strip() != '')

class RsssuckerFilter():
    '''
    filter that filters out small (less than 40 tokens) and empty texts'
    40 alphanum tokens boundary is determined by short text sample inspection.
    '''

    def __init__(self):
        self.tokenizer = prefilter_tokenizer()

    def getId(self): return 'filter_lt40tok'

    def __call__(self, txtobj):
        if len(txtobj.text) == 0 : return True
        if len(self.tokenizer.tokenize(txtobj.text)) < 40  : return True
