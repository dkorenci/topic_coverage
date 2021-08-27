from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics.cluster import silhouette_score
from doc_topic_coh.mixedness.ClusteringMixedness import ClusteringMixedness

def kmeansSilhouette(init='k-means++'):
    cl = KMeans(init=init,
        n_clusters=2, random_state=235, n_jobs=3)
    score = silhouette_score
    return ClusteringMixedness(cl, score)

def spectralSilhouette(affinity, n_neighbours=10, kernel_params=None):
    cl = SpectralClustering(affinity=affinity, n_neighbors=n_neighbours,
                            kernel_params=kernel_params,
                            n_clusters=2, random_state=566, n_jobs=3)
    score = silhouette_score
    return ClusteringMixedness(cl, score)

def hacSilhouette(linkage, affinity):
    cl = AgglomerativeClustering(linkage=linkage, affinity=affinity,
                                 n_clusters=2)
    score = silhouette_score
    return ClusteringMixedness(cl, score)

if __name__ == '__main__':
    mix = spectralSilhouette(affinity='cosine')
    print mix.id