Package with the resources for the experiment described in
"Getting the agenda right: measuring media agenda using topic models"
as well as additional resources derived later.
Resources are wrapped as pytopia objects and packaged in pytopia contexts.

Basic resources include:
corpora, dictionaries and text2tokenizers (text preprocessors)

Corpora texts are assumed to be stored stored in a local postgresql database
defined and created by the feedsucker app (https://github.com/dkorenci/feedsucker).

Derived resources are:
artificial topic model with topic corresponding to semantic topics (themes)
that are derived from annotated lda model topics