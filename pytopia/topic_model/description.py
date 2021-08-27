from lxml import etree
from lxml import objectify

import os
'''
Saving and loading of topic descriptions to/from XML file
'''

def test():
    xml_desc = ''.join(open('model_desc.xml').readlines())
    print xml_desc
    o = objectify.fromstring(xml_desc)
    return o

def load_from_file(fileName):
    file = None
    try:
        file = open(fileName)
        xml_desc = ''.join(file.readlines())
        file.close()
        o = objectify.fromstring(xml_desc)
        return o
    finally:
        if file is not None: file.close()

def save_description_to_file(fileName, desc):
    file = None
    try:
        xml_desc = to_xml(desc)
        file = open(fileName, 'w'); file.write(xml_desc)
        file.flush()
        os.fsync(file.fileno())
        file.close()
    finally:
        if file is not None: file.close()


def decription_from_model(model):
    'create empty decription for TopicModel'
    d = empty_description()
    try : d.model_id = model.id()
    except: d.model_id = ''
    for i in model.topicIds() :
        t = empty_topic()
        t.topic_index = i
        t.label = 'Topic '+str(i)
        d.append(t)
    return d

def empty_description(numTopics = None):
    'create empty description ObjectifiedElement'
    xml = '''
        <model_description>
            <id></id>
            <model_id></model_id>
            <description></description>
        </model_description>
    '''
    d = objectify.fromstring(xml)
    if numTopics is None: return d
    #todo enable setting of custom topic indices
    for i in range(numTopics) :
        t = empty_topic()
        t.topic_index = i
        t.label = 'Topic '+str(i)
        d.append(t)
    return d

def empty_topic():
    'create empty topic ObjectifiedElemet'
    xml = '''
        <topic>
            <topic_index></topic_index>
            <label></label>
            <description></description>
        </topic>
    '''
    return objectify.fromstring(xml)

def to_xml(object):
    objectify.deannotate(object, cleanup_namespaces=True)
    return etree.tostring(object, pretty_print=True)
