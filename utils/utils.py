import codecs, string
from textwrap import wrap
import os, psutil
from multiprocessing import cpu_count

# set process affinities (attachment to particular cores) to all cores
def unpinProcess():
    p = psutil.Process(os.getpid())
    p.set_cpu_affinity(range(cpu_count()))

def normalize_path(folder):
    'add / to the end of path, if missing'
    if len(folder) > 0 :
        if folder[-1] != '/' : folder = folder + '/'
    return folder

def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]

def print_text_objects(file, texts, ordinals = False):
    'print a list of Text objects to a file'
    f = codecs.open(file, "w", "utf-8")
    cnt = 0
    for txto in texts :
        cnt+=1
        if ordinals: f.write('ordinal: %d'%cnt + '\n');
        f.write('id: '+str(txto.id)+'\n')
        f.write('title: '+txto.title+'\n')
        f.write(wrap_text(txto.text, 100)+"\n\n")
        f.write("****************************************************\n\n")

def wrap_text(text, charsPerLine):
    return '\n'.join( wrap( string.replace(text, '\n', ' ') , charsPerLine ) )