# ML-engineer


| Folder                                      | Project                                      | Status       |
|---------------------------------------------|----------------------------------------------|-------------|
| /GPT-2                                      | **GPT-2**, replicating GPT2 from scratch following [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) but with my own tokenizer | 🟠 **In Progress** | 
| /BERT                                  | **BERT**, replicating BERT paper (without additional guidance) using my own tokenizer | ✅ **Completed** | 
| /llm-tokenizer                                      | **Tokenizers**, LLMs tokenizers following [karpathy/minbpe](https://github.com/karpathy/minbpe) | 🟠 **In Progress** | 
| /mingpt                                     | **MiniGPT**, implementing transformers from 'Attention is All You Need' following [karpathy/minGPT](https://github.com/karpathy/minGPT) | ✅ **Completed** |
| /makemore                                   | **Makemore**, simple language models following  [karpathy/makemore](https://github.com/karpathy/makemore) |  ✅ **Completed** |
| /micrograd                                  | **Micrograd**, implementation of backpropagation following [karpathy/micrograd](https://github.com/karpathy/micrograd) |  ✅ **Completed** |
| /zero-shot-retrieval                        | **Zero Shot LLM Retrieval**, submissions to [Kaggle VMWare Zero-shot Retrieval competition](https://www.kaggle.com/competitions/vmware-zero-shot-information-retrieval) |  ✅ **Completed** |
| /personalized-fashion-recommendations       | **SparseNN Recommender System**, submissions to [Kaggle H&M Recommender System competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) |  ✅ **Completed** |
| /algorithms                                 | Codeforces contests and Leetcode Hard Design questions | 🟠 **In Progress** |


-----------

## ML-engineer/GPT-2

Replicating GPT2 from scratch following [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) but with my own tokenizer. This is an example of improving token/s throughput for given model size by optimizing pytorch operations. Read more on my [GPT2 training notes](https://docs.google.com/document/d/12R4rATOytr2OkNysWoisq6W2CX2fWkpWQcm3G8LRyE4/edit?tab=t.0)

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/gpt2-throughtput1.png)




## ML-engineer/BERT

Implemention of BERT paper without additional guidance using my own tokenizer. This is an example of pretraining on Next Sentence Prediction (NSP) and Masked Language Modelling (MLM) tasks.

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/bert-losses2.png)

## ML-engineer/MiniGPT

Implementation of "Attention is All You Need" trasfomer architecture with minimal pytorch APIs, similar to [karpathy/minGPT](https://github.com/karpathy/minGPT). This is the next word prediction cross-entropy loss achieved on the Shakespeare dataset with different baselines.

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/minGPT-losses.png)

*the number of parameters is wrong, should be in the range of millions. Re-running the best model with Karpathy hyperparams achieved a validation loss of 1.66. This is an example generation of Shakespeare like text with Transformer@3k parameters. 
![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/minigpt-shakespeare1.png)


## ML-engineer/Makemore

Replicated the makemore repo from Karpathy from his Zero to Hero NN course. Implementation of name-generation language models. Bi-grams, MLP, RNNs and other models in plain pytorch. This is the performance I was able to reproduce independently on the several architectures covered in the course.

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/makemore-performance.png)

Here are some interesting histogram from hyperparameter search on some simple language model

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/makemore-hist1.png)
![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/makemore-hist2.png)

## ML-engineer/Micrograd

Python-only implemention of Neural Networks. Playing with my own implementation of [micrograd](https://github.com/karpathy/micrograd) from Karpathy. Some interesting results

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/micrograd-MLP.svg)

- [make_moons_30_Jan_A.ipynb](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/micrograd/make_moons_30_Jan_A.ipynb) - a small MLP is able to optimize loss function, but it learns a linear function. Not able to make the model learn non linearity.
- [make_moons_30_Jan_B.ipynb](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/micrograd/make_moons_30_Jan_B.ipynb) - a small MLP with more techniques is able to learn non linear function from scikit learn moon. The circles function are half learned but not completely 

![](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/imgs/micrograd-1.png)
![](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/imgs/micrograd-2.png)


## ML-engineer/Zero Shot LLM Retrieval

Using VMWare docs corpus (30M documents) from Kaggle to implement a e2e retrieval system using LLM encoders and generative models. Picture below is the tensorboard of the 12 stacked transformer blocks from https://huggingface.co/intfloat/e5-small-v2 used for text embedding

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/zero-shot-retrieval.png)


## ML-engineer/SparseNN Recommender System 

Using Fashion Recommender System dataset to build a muli-stage ranking recommender system for 10M users and 100k fashion articles [https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

- [personalized-fashion-recommendation-2-Feb-B.ipynb](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/personalized-fashion-recommendations/personalized-fashion-recommendation-2-Feb-B.ipynb) TTSN model for candidate retrieval, trained only on categorical features with customer and article tower, improving recall@1000 from 1% to 10%. Will probably need to bring recall higher before moving on to ranking stage.

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/recommendersystem-recall1.png)





## Other Resources

- https://github.com/alirezadir/machine-learning-interviews/blob/main/src/MLSD/ml-system-design.md	
- https://pytorch.org/tutorials/intermediate/torchrec_intro_tutorial.html 
- https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022 
- https://web.stanford.edu/class/cs246/slides/07-recsys1.pdf 
- http://cs246.stanford.edu/ 
- Tue Jan 28	Recommender Systems I
- [slides]	
- Ch9: Recommendation systems
- Thu Jan 30	Recommender Systems II
- [slides]	
- Ch9: Recommendation systems
- https://applyingml.com/resources/discovery-system-design/ 
- https://applied-llms.org/ 
- Neel Nanda Transformers https://www.youtube.com/watch?v=bOYE6E8JrtU&list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz&index=1
