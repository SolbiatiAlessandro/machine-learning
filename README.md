# ML System Design

Repo to prepare for ML System Design interview. 

- [Not Started] Recommender System (Personalized Fashion Recommendations)
- [Not Started] Search System challenge
- [In Progress] NanoGPT
- [Completed] Micrograd

-----------

## [In Progress] Recommender System (Personalized Fashion Recommendations)

Using Fashion Recommender System dataset to build a muli-stage ranking recommender system for 10M users and 100k fashion articles [https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

- [personalized-fashion-recommendation-2-Feb-B.ipynb](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/personalized-fashion-recommendations/personalized-fashion-recommendation-2-Feb-B.ipynb) TTSN model for candidate retrieval, trained only on categorical features with customer and article tower, improving recall@1000 from 1% to 10%. Will probably need to bring recall higher before moving on to ranking stage.

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/recommendersystem-recall1.png)

## [Not Started] Search System challenge

I want to find a good search system design dataset to implement exercise e2e search system.

## [In Progress] NanoGPT

NN Zero to Hero from Karpathy. Implementation of GPT-2 from scratch in pytorch. [https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)

- [nanogpt-mlp.ipynb](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/nanogpt/nanogpt-mlp.ipynb) - interesting experience where a simple MLP is able to get 50% accuracy on tiny-shakespeare dataset and spit out almost coherent sentences (the prompt is highlighted in light blue in images below)

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/nanogpt-mlp1.png)
![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/nanogpt-mlp2.png)
![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/nanogpt-mlp3.png)

## [Completed] Micrograd

Python-only implemention of Neural Networks. Playing with my own implementation of micrograd from Karpathy. Some interesting results

![](https://raw.githubusercontent.com/SolbiatiAlessandro/ML-system-design/refs/heads/main/imgs/micrograd-MLP.svg)

- [make_moons_30_Jan_A.ipynb](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/micrograd/make_moons_30_Jan_A.ipynb) - a small MLP is able to optimize loss function, but it learns a linear function. Not able to make the model learn non linearity.
- [make_moons_30_Jan_B.ipynb](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/micrograd/make_moons_30_Jan_B.ipynb) - a small MLP with more techniques is able to learn non linear function from scikit learn moon. The circles function are half learned but not completely 

![](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/imgs/micrograd-1.png)
![](https://github.com/SolbiatiAlessandro/ML-system-design/blob/main/imgs/micrograd-2.png)


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
