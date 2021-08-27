This package contains the code and data of the measures
and the experiments described in the paper "A Topic Coverage Approach to Evaluation of Topic Models",
authored by Damir Korenčić, Strahil Ristov, Jelena Repar, and Jan Šnajder

The code is licensed under the Apache 2.0 License

Detailed documentation of the data and the code, including setup instructions
and pointers from paper sections and tables to the corresponding code, can be found in
data.readme.txt and code.readme.txt.


******** A note on the pytopia framework ********

The inherent complexity of the experiments is due to the need
to build and manage a large number of topic models and related resources, 
and to calculate various evaluation functions that take these models as input.

These problems are solved via the pytopia framework (contained in the pytopia package), 
which provides interfaces for topic models and other standard resources, 
and is based on identifiability and hierarchic compositionality.
This means that every object (such as topic model) has an unique string id, 
and is composed of or derived from other lower-level objects and resources.
The ids of every object must be constructed to reflect these dependancies.
For example a topic model depends, at the very least, on a tokenizer and a text corpus.

The ubiquitious identifiability of objects enables referencing and caching of both built resources and function values.
The caching functionality underlies most of the experiments and enables re-use of built resources,
quick re-runs of calculated function, and reproducibility -- we distribute both the 
cached resources and the calculated function values with our experiments.

Another feature of pytopia is the use of resource contexts that enable 
packaging of and access to the various resources via string ids. 
For example, if a method creating the context is called createContext(), 
then the code using the in-context resources would look like this:

with createContext():
    my_resource = resolve('resource_id')
    useResource(my_resource)


******** How to run coverage experiments on your own model? ********

First adapt your model to the TopicModel interface (pytopia.topic_model.TopicModel), 
an example is the SklearnNmfTmAdapter class (pytopia.adapt.scikit_learn.nmf.adapter)
The constructor should accept as params all the relevant resources and hyperparameters.

Next, use the resource context that contains corpora, tokenizers, etc., of our experiments, 
located in: topic_coverage.resources.pytopia_context.topicCoverageContext
Make sure to setup the locations for storing intermediate resources,
by setting the settings.resource_builder_cache variable.
For more info, take a look at: topic_coverage.resources.pytopia_context.builderContext, and data.readme.txt

Then build the instances of your topic models, as exemplified 
by the code in: topic_coverage.modelbuild.modelbuild_docker_v1

Finally, adapt and use the method for calculating coverages: 
topic_coverage.experiments.coverage.experiment_runner.evaluateCoverage()
The adaptation is performed by creating and using a custom method for
loading your own models, that can be created by adapting this method: topic_coverage.modelbuild.modelset_loading.modelset1Families()

Finally, you will want to setup or build the supervised matcher, 
and the function caches (for experiment re-running).
More info on this can be found in code.readme.txt

