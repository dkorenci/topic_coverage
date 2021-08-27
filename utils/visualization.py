import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
from scipy.stats import entropy

def visualize_labeled_topic(topic, axes = None, label = False, topN = 20):
    "draw  bars for word probabilities in the topic"
    values, words = top_words(topic, topN)
    if axes is None: fig, axes = plt.subplots(1, 1)
    xcoords = np.arange(topN)
    axes.xaxis.set_major_locator(ticker.FixedLocator((xcoords)))
    if label: # label x axis with words
        axes.xaxis.set_major_formatter(ticker.FixedFormatter((words)))
        labels = axes.get_xticklabels()
        plt.setp(labels, rotation = 90.)
    else:
        plt.setp(axes.get_xticklabels(), visible=False)
    axes.bar(xcoords, values, align = 'center', color = '.75')
    axes.margins(x=0,y=0)

def top_words(topic, topN = 20):
    'get top topN words by probability, and their labels'
    vec = topic.vector
    if len(vec) < topN: topN = len(vec)
    top_indices = np.argsort(vec)[::-1][:topN] # get indices in sorted order, reverse, take first topN
    values = [ vec[i] for i in top_indices ] # get topN values
    words = [ topic.model.dictionary[i] for i in top_indices ]
    return values, words

def top_vector_indices(vec, topN):
    if len(vec) < topN: topN = len(vec)
    top_indices = np.argsort(vec)[::-1][:topN] # get indices in sorted order, reverse, take first topN
    return top_indices

def draw_labeled_bars(values, labels = None, axes = None, xtickFontsize = None):
    "draw  bars for word probabilities in the topic"
    if axes is None: fig, axes = plt.subplots(1, 1)
    xcoords = np.arange(len(values))
    axes.xaxis.set_major_locator(ticker.FixedLocator((xcoords)))
    if labels is not None: # label x axis with words
        axes.xaxis.set_major_formatter(ticker.FixedFormatter((labels)))
        plt.setp(axes.get_xticklabels(), rotation = 270.)
        if xtickFontsize is not None:
            for tick in axes.get_xticklabels(): tick.set_fontsize(xtickFontsize)
    else:
        plt.setp(axes.get_xticklabels(), visible=False)
    axes.bar(xcoords, values, align = 'center', color = '.75')
    axes.margins(x=0,y=0)

def entropy_distribution(model, axes = None):
    "box-and-whiskers plot for topic entropies"
    if axes is None: fig, axes = plt.subplots(1, 1)
    ents = []
    # add topic entropies
    for i in range(model.num_topics):
        ents.append(entropy(model[i].vector))
    # add minimal and maximal entropy for this number of topics
    num_words = len(model[0].vector)
    min = np.zeros(num_words); min[0] = 1. ; ents.append(entropy(min))
    max = np.ones(num_words); ents.append(entropy(max))
    axes.boxplot(ents)