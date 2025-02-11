kaggle competition https://www.kaggle.com/competitions/vmware-zero-shot-information-retrieval

## Problem Requirement

- real queries from our Google Search Console data, searching our highly-technical corpus of documentation
- 2.3k queries
- 325k corpus
- find 5 documents that best answer each question
- no lexical search queries: We've limited the query set to queries that more or less resemble questions because they are far more challenging. Any old keyword-based search engine can find the correct document for "vSphere 7.0 release notes", but will fall flat when faced with something like, "why is everyone doing this isolate x copy and paste".
- domain specific information retrieval vs general information retrieval
- zero-shot, no labelled training set
- evaluation metric nDCG@5
- return 5 documents id for each query id
- duplicated documentation (GUID)

## Dataset

- document group: blog, cb, glossary, sdw, website, documentation..
- filename, filetext, filemetadata

## Evaluation

- how are we going to evaluate this thing? pretty tricky
- in prod environment you can check user click and use that as a relevance/gain score
- we can use LLM as a judge
- 600M pairs sampled, get a relevance score, and get nDCG

## Data
- we have different document type, we shuold have different treatemetns
- what's in metadata

doc: metadata has keywords, description
- text length
- how to split in chunk, different chunk strategies

doing some EDA on data
### Documents
mostly documents and blogs, each one has some extraction in metadata
30k Blogs length between 1k and 20k characters
200k docs lengths between 1k and 5k characters
queries

### Queries
lot to think about queries
queries matter a lot
15% what's going on here is people having errors while using vmware systems
25% if what is <> queries
60% is how to <> queries, they are pretty simple queries 


## Modelling

baseline would be frequency of keyword match


- embedding model
- bi-encoder (same encoder for text and query -> sentence BERT)
- different encoders, query encoder, and document encoder
- embedding model trained with contrastive loss, heavier but more robust

- multistage ranking 
first figure out the area
and then do more closer tuning

prototype is take out of the box embedding model
embed all the documents
overlapping paragraph -> back to document
send user query to LLM and ask LLM to make several queries
embed the query and get KNN from documents embeddings (bi-encoder)
get top 100 paragraphs
for each paragraph do a cross encoder for re ranking
can the cross encoder be a LLM ?


