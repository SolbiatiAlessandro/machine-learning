notes and implementations from the book [https://github.com/HandsOnLLM/Hands-On-Large-Language-Models?tab=readme-ov-file](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models?tab=readme-ov-file)

# Chapter 8.1 Dense Retrieval 
## Considerations for dense retrieval

## Requirements
1. what if answer is not in the text?
3. both semantic and keyword search if user thinks is a keyword search

## Chuncking
4. what if answer is in multiple sentences?

Different chunking strategies
- sentence/paragraph
- character split (every x characters)
- token split (every x tokens)
- token split with overlapping tokens
- LLM to dynamically split text into meaningful chunks

## Labels
2. track user clik on answer can be use to improve future versions (how)
5. thinking about labels -> we need to know if the user found the answer or not, if I am asking Gemini for instance is unclear whether the answer was helpful or not. Thumbs up and down are a good start.

## Indexing
- indexes like FAISS where they can tradeoff precision for speed, for instance clustering the indexes. But the indexes have to be trained.
- using vector DB allows you to add a new vector without having to rebuild the index

## Fine Tuning
- fine tune embedding for retrieval
- have retrieval queries and fine tune the LLM to optimze for embedding retrieval


## Example of retrieval datasets

> https://www.kaggle.com/competitions/vmware-zero-shot-information-retrieval?utm_source=chatgpt.com

> https://github.com/project-miracl/miracl

> https://arxiv.org/pdf/2112.09118

example of informationl retrieval kaggle competition

------------
# Chapter 8.2 ReRanking 

## ReRanking

- first stage, keyword/lexical search like BM25
- second reranking, **cross-encoder architecture** via query-document pairs are assigned a score. This is a classification problem.

1. https://arxiv.org/abs/1910.14424 Multi-Stage Document Ranking with BERT
2. https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/retrieve_rerank

## ReRanking Evaluation

- **Mean Average Precision**: average P@K for all the relevant document in the final rank, averaged across different query. Balance between precision and recall
- **nDCG** (non discounted cumulative gain) is for non binary classification. Gain is the relevance of document at position k, where is not just binary but there is a relevance score. We discount it by its position (/k) and we normalize it.

1. https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html


# Chapter 8.3 RAG

1. https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks
2. Embedding Model is trained on contrastive loss, LLM is trained on cross-entropy next work prediction
3. Advanced RAG techniques

- **Query Rewriting**: use the LLM to rewrite the query into one that aids the retrieval step. User Question -> LLM question -> RAG
- **MultiQuery RAG**: have several RAG questions
- **Query Routing**: different retrieval system
- **Agentic RAGs**: query sources become APIs

# RAG Evaluation

- https://arxiv.org/pdf/2304.09848 Evaluating Verifiability in Generative Search Engines: not all generated statements are
fully supported by citations (**citation recall**), and not
every citation supports its associated statement (**citation
precision**).
- **LLM-as-a-judge**:  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ capable LLM acting as a judge to score the generation on different axis like faithfulness (answer consistent to provided context) and relevance
