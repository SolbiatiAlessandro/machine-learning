# Text Classification

Following [Hands On LLM - Text Classification chapter](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter04/Chapter%204%20-%20Text%20Classification.ipynb#scrollTo=X0KyKHtqyjn3) to evaluate different LLM-based classifiers on Movie Review Sentiment Classification task from rotten tomatoes dataset. Here are some findings

| Base Model         | Features               | Classifier         | Accuracy |
|--------------------|------------------------|--------------------|----------|
| Random Initialized (SmallBERT) | CLS of first 80 tokens | Linear Regression  | 0.57     |
 Random Initialized (BigBERT) | CLS of first 80 tokens | Linear Regression  | 0.58     |
| SmallBERT          | CLS of first 80 tokens | Linear Regression  | 0.55     |
| BigBERT          | CLS of first 80 tokens | Linear Regression  | 0.58    |
|--------------------|------------------------|--------------------|----------|


SmallBERT is a 20M params BERT implemented and trained from scratch on tiny shakespeare for 5k epochs (~30mins on 1 GPU)
LargeBERT is a 20M params BERT implemented and trained from scratch on wikitext2 for 150k epochs (~6h on 1 GPU)

You can see model implementaiton at ../BERT
