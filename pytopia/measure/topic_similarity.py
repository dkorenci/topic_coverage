from math import exp

# distance to similarity converters
def invert(dist): return 1.0 / (1.0 + dist)
def einvert(dist): return 1.0/exp(dist)

from pytopia.measure.topic_distance import cosine as cosineDist
def cosine(m1, m2): return 1.0 - cosineDist(m1, m2)

from pytopia.measure.topic_distance import jensenShannon as jsDist
def jensenShannon(m1, m2): return -jsDist(m1,m2)

from pytopia.measure.topic_distance import l2 as l2Dist
def l2negative(m1, m2): return -l2Dist(m1, m2)


