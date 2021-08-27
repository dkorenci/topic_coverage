def topicPid(topic):
    ''' Return topic's "pair id" consisting of model id and in-model topic id. '''
    return (topic.model, topic.topicId)

def pid2Topic(pid):
    ''' Create using pytopia context a Topic object for (modelId, topicId) pair'''
    from pytopia.context.ContextResolver import resolve
    mid, tid = pid
    model = resolve(mid)
    return model.topic(tid)
