# Natural Language Searching with p-means sentence embeddings

## Intro
This extension of the Flask API from the UKPLab p-means sentence embeddings project, takes advantage of the existing model (with 'monolingual' English embeddings enabled by default), to provide a function for searching over natural language descriptions (for example, of products), by comparison of p-means sentence embeddings vectors (specifically by measuring their cosine distance to some input search term, from which a new embedding is generated).

This makes for accurate semantic natural language searching for niche/complex application domains.

Currently the service loads in 'nomenclature' (descriptions) for a given set of items from an Excel file hosted in Azure Storage. However, you may change this however you wish, ultimately providing a dictionary of descriptions from which we generate embeddings on which a search is based. For speed of search, these are generated as the web service loads on deployment/testing. Note that this does result in the drawback that the service is currently slow to boot. 

This is very basic and quickly implemented, but has proved powerful in testing. All credit to UKP Lab for providing the underlying technology and template API. 

Written in Python 2. A Python 3 translation would be a great contribution. :)

(More detail w/ code snippets incoming.)