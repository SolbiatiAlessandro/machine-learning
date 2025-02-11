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

embedding process
- start with some sampled data
- we should do per document type embedding but for now
- let's keep that as hyperparameters

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


## Notes Feb 10

Able to get embeddings using BERT-like model https://huggingface.co/intfloat/e5-small-v2

T2 GPU inference time + memory
- 300MiB to load the model weights
- 1315MiB /  15360MiB after first inference of 300
- 2709MiB /  15360MiB inference of 600
- 13913MiB /  15360MiB  inference for 3000 chunks, takes 3 seconds (3 * 10^3 seconds)
- total number of chunks is 3*10^7 = 10^4 seconds to embed them all = 3 hours
- I have two GPUs and I can do it in parallel and is going to take 1 hour

CPU memory storage
- 3000 embeddings are 3000 * 348 emb size * 4bytes = 1MB in memory  = 10^6
- I have 10GB memory 10^10 , can hold 3 * 10^3 * 10^4 = 3 10^7 embeddings

Next steps need to setup a end to end evaluation loop to iterate before starting the e2e embedding

steps missing
- indexing of the embedding
- query rewriting from LLM
- get top X , use LLM to judge relevance score, compute nDCG@5

questions
- how to sample for the e2e loop? could search for a specific keywords the docs are talking about to make sure that all queries and all documents are about the same topic
- optional would be to use a cross-encoder model to re-rank top X
