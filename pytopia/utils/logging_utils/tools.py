import os
from os import path
from shutil import copyfile, copytree

def fullClassName(obj):
    return obj.__module__ +'.'+ obj.__class__.__name__

def logFile(file, newFname=None):
    '''
    Copy file to folder with name and location corresponding to
    those of root logger file, optionally under new name.
    '''
    from pytopia.utils.logging_utils.setup import rootLoggerFolder
    if rootLoggerFolder is None: return False
    try:
        if not path.exists(rootLoggerFolder): os.mkdir(rootLoggerFolder)
        fname = newFname if newFname else path.basename(file)
        newfile = path.join(rootLoggerFolder, fname)
        if path.isdir(file):
            copytree(file, newfile)
        else:
            copyfile(file, newfile)
    except: return False
    return True