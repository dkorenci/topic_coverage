from pytopia.resource.loadSave import loadResource, saveResource
from pytopia.utils.logging_utils.setup import *
from pytopia.utils.logging_utils.tools import fullClassName

from os import path
from os.path import basename, normpath
import cPickle, re, gc

class ResourceBuilderCache():
    '''
    Cache resources by saving them in folders named after hashed ids.
    Similar to a set implemented with hashing and array of linked lists.
    Currently based on code of obsolete FolderResourceCache.
    TODO: safe concurrent access to cache
    '''

    def __init__(self, builder, folder, id=None, memCache=True,
                        dummyHash=False):
        '''
        :param memCache: if True, also cache built/loaded object in memory
        :param dummyHash: if True, hash function will be very simple and cause frequent
                hash-crashes, used only for testing this eventuality
        '''
        self._builder = builder
        self._folder = folder
        self._log = createLogger(fullClassName(self), INFO)
        self.id = id
        self._memCache = memCache
        self._dummyHash = dummyHash
        if memCache: self._cache = {}

    def _hid(self, id):
        '''
        Hash of resource ids reproducible across runs and machines,
            suitable to be used as a folder name.
        '''
        if self._dummyHash:
            d = sum(ord(c) for c in str(id)) % 2
            return 'hid%d' % d
        from hashlib import pbkdf2_hmac
        h = pbkdf2_hmac('sha512', str(id), 'hashed_resource_cache', 10000, dklen=50)
        hid = 'hid'+''.join('%d'%ord(b) for b in h)
        return hid

    @staticmethod
    def _isHidFolder(folder):
        '''
        True if folder matches naming convention for resource hash defined in _hid
        '''
        return re.match('hid[0-9]+', basename(normpath(folder)))

    def resourceId(self, *args, **kwargs):
        return self._builder.resourceId(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        id = self._builder.resourceId(*args, **kwargs)
        # try memcache
        if self._memCache:
            if id in self._cache:
                self._log.info('mem-cache retrieve success: %s' % id)
                return self._cache[id]
        # load from disk cache, or build resource
        cached = self._diskCacheLookup(id)
        if cached:
            self._log.info('retrieve success: %s' % id)
            if self._memCache: self._cache[id] = cached
            return cached
        else:
            self._log.info('retrieve fail, building: %s' % id)
            res = self._builder(*args, **kwargs)
            self._saveToDiskCache(res)
            if self._memCache: self._cache[id] = res
            return res

    @staticmethod
    def loadResources(folder, filter=None, asContext=True, loadMock=None):
        '''
        Load all resources stored in the folder via ResourceBuilderCache.
        :param asContext: if True return pytopia Context, otherwise return list
        :param filter: if not None, only resources with ids matching this regex will be returned
                regex is either a regex string or a list in which case regex will correspond
                to elements of the list with any characters before, after and in between.
        :param loadMock: if not None, must be a class that supports creation of mock objects.
                in this case, mock objects are created from loaded objects and returned.
        '''
        from pytopia.utils.file_utils.location import FolderLocation as loc
        from pytopia.resource.loadSave import loadResource
        import re
        if isinstance(filter, list):
            filter='.*'.join(re.escape(f) for f in filter)
            filter = '.*'+filter+'.*'
        res = []; numUnsavedRes = 0; collectEvery = 20
        for fi in loc(folder).subfolders():
            if ResourceBuilderCache._isHidFolder(fi):
                for fj in loc(fi).subfolders():
                    if ResourceBuilderCache._isResFolder(fj):
                        r = loadResource(fj)
                        if filter and not re.match(filter, str(r.id)): numUnsavedRes += 1
                        else:
                            if loadMock:
                                print r.id
                                mockres = loadMock()
                                mockres.copyMock(r)
                                res.append(mockres)
                            else: res.append(r)
                    if numUnsavedRes and numUnsavedRes % collectEvery == 0:
                        gc.collect()
                        numUnsavedRes = 0
        if numUnsavedRes: gc.collect()
        if asContext:
            from pytopia.context.Context import Context
            ctx = Context('')
            for m in res: ctx.add(m)
            return ctx
        else: return res

    _bucketIdsFile = 'id2index.pickle'
    def _diskCacheLookup(self, id):
        '''
        :return: cached resource or None
        '''
        f = self._folderById(id)
        ids2index = self._loadIds2Ind(f)
        if id in ids2index:
            f = self._folderById(id, ids2index[id])
            return loadResource(f)
        else: return None

    def _saveToDiskCache(self, res):
        '''
        Save resource if it does not exist in cache,
            first check id -> bucket-level-index map in the hash-bucket folder,
            then if the resource is not in the map, create new index,
            update map, create corresponding subfolder, and save.
        :param res: pytopia resource
        '''
        f = self._folderById(res.id)
        ids2index = self._loadIds2Ind(f)
        if res.id in ids2index:
            # this could also be a warning, but for treat as error
            #  cause saving should not happen if resource is already present
            msg = 'saving already present resource with id: %s' % str(res.id)
            self._log.error(msg)
            raise Exception(msg)
        else:
            newIndex = max(ids2index.values())+1 if ids2index else 0
            ids2index[res.id] = newIndex
            saveResource(res, self._folderById(res.id, newIndex))
            self._saveIds2Ind(ids2index, self._folderById(res.id))

    def _id2IndFname(self, folder):
        return path.join(folder, self._bucketIdsFile)

    def _loadIds2Ind(self, folder):
        '''
        Load pickled resource.id -> bucket-level-index map for the bucket folder.
        If the saved map does not exist, return empty map.
        '''
        fname = self._id2IndFname(folder)
        if os.path.exists(fname):
            f = open(self._id2IndFname(folder), 'rb')
            return cPickle.load(f)
            f.close()
        else: return {}

    def _saveIds2Ind(self, ids2ind, folder):
        f = open(self._id2IndFname(folder), 'wb')
        cPickle.dump(ids2ind, f)
        f.close()

    def _folderById(self, id, index=None):
        '''
        Given resource's id, return path to folders where resource is to be saved.
        If index is None, folder corresponding to hash bucket is returned,
          else subfolder of cached bucket corresponding to index is returned.
        '''
        basepath = path.join(self._folder, self._hid(id))
        if index is None: return basepath
        else:
            indSubDir = 'res%d'%index
            return path.join(basepath, indSubDir)

    @staticmethod
    def _isResFolder(folder):
        '''
        True if folder matches naming convention for resource folders defined in _folderById
        '''
        return re.match('res[0-9]+', basename(normpath(folder)))


if __name__ == '__main__':
    pass