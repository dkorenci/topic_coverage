tfidf_matrices_folder = '/data/resources/phenotypes/tf-idf matrices/all but one/'
corpusLabels = [ 'hamap', 'microbewiki', 'PMCArticles', 'pubmedSearch', 'wikipedia' ]

# xlsx table and sheet containing definition of phenotype topics
phenoTopicsLocation = ('/data/resources/phenotypes/derived/phenotype_clusters.xlsx',
                  'NMF_factor groups')

# folder with txt files containing NMF topics (clusters) selected as phenotypes
origClustersFolder = '/data/resources/phenotypes/clustersSelectedToKeep_FS/'

# folder with files containing word-bacteria weights, for each cluster and corpus
topicBacteriaFolder = '/data/resources/phenotypes/final clusters doctopics/'

# file with mapping phenotype handle (top words) -> id of the cluster (folder+clusterId)
pheno2clusters = '/data/resources/phenotypes/final clusters doctopics/NMF_clusters_mapping'
