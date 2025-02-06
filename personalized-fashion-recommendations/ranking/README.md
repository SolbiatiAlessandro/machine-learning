# sparsenn_linearlayer.ipynb

We implement embedding lookup table with only one fully connected linear layer. As expected, the linear layer is not able to learn any informations and converges to random guess loss.

Random guess loss correspond to 50% guess as training data are balanced, if we plug it in the cross entropy loss we get `-log(0.50)*1 - log(0.50)*0 = 0.69`. The training and test losses convege to 0.69. We also verify that the weights gets pushed to 0.00 by the training process to achieve completely random predictions.

![](https://github.com/SolbiatiAlessandro/ML-system-design/blob/344a3404847cdf924e91722df972708bdc820770/personalized-fashion-recommendations/ranking/sparsenn_linearlayer.png)
