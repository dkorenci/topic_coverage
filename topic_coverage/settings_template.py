# generic resource folder, not used in main experiments
resource_folder  = None
# folder for storing built pytopia resources
resource_builder_cache = None
# folder for storing cached results of functions
function_cache_folder = None

# folder containing the topic models
topic_models_folder = None

# annotated topic pairs of the biological dataset
# !!! convert to absolute path for the entry in settings.py
phenotype_pairs_folder = '../../data/labeled_topic_pairs/bio_topic_pairs'
# annotated topic pairs of the news dataset
# !!! convert to absolute path for the entry in settings.py
uspol_pairs_folder = '../../data/labeled_topic_pairs/news_topic_pairs'
# subset of models used for sampling topic pairs
topicpairs_sample_models = None

# supervised matching models
supervised_models_folder = None

# models for measuring size of ref. topics
topicsize_measure_models = None

# annotations of semantic categories of news ref. topics
news_semantic_annot = None

# palmetto indices, for coherence measures
palmetto_indices = None

# stability mock models
stability_mock_models = None
# folder containing the extended set of topic models (stability experiments)
topic_models_extended = None
# function cache for the extended/mock set of models
stability_function_cache = None
