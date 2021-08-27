from codecs import open as copen

class Text(object):
    '''
    abstract imput for text processing, a text with associated id
    '''
    
    def __init__(self, id, text, title = None):
        self.text, self.id = text, id
        self.title = title

    @staticmethod        
    def fromFile(file):
        '''
        read Text object from a file: 1st line id, 2nd line title, the rest is text
        '''
        lines  = [l.rstrip() for l in copen(file,'r',encoding='utf-8').readlines()]
        id = lines[0] if lines[0] != '' else None 
        title = lines[1]
        t = '\n'.join(lines[2:])
        text = Text(id, t); text.title = title
        return text
    
    def toFile(self, fname):
        '''
        write Text object to file, so that it can be read by fromFile
        '''
        file = copen(fname, "w", encoding='utf-8')
        file.write(str(self.id) + "\n")
        try: file.write(self.title + "\n")
        except AttributeError: file.write("\n");
        file.write(self.text)
        file.close()    
            