# Problem Framing

Recommendation problem

customer -> [p1, p2, p3..]

maximise likelihood of customer buying the product.

different objective: viewing, adding to cart, purchase

<user, product> - onsite p(view), p(click), offsite p(purchase), 

p(purchase)
score = view_weight * p(view) + purchase_weight * p(purchase)

in this case we have only purchase but building a MTML model can help to avoid overfitting, data sparsity and correlation between labels that share same representation.

negative label, didn't click. p(scroll fast), p(not interested)

multistage ranking
- retrieval 10^5 -> 10^2, lighter models
- ranking 10^2 -> 10, heavier models

offline metrics (precision, recall), NE 
online metric - click, purchase click

# Training Data

lables 

class imbalance

- **cold start, new user id won't have any data from before**
- logging, data cleaned. 
- **Online and offline coverage**
- negative sampling, strategy. Random sampling, more sophisticated like popularity, Screen passed, X-s out.

article_id, prod_name, prod_type,name, product_group, graphical appearance no/name, colour

color -> user preferences user x color features

customers
customer_id, club member status, fashio news frequency, age , postal code

postal code -> wealth
average postal code spending

images 
- color, more detailed product groups, unsupervised clustering within groups,


# Feature Engineering

user features
- info
- history, 7 days purchases, 30 days purchases
- postal code 7 days purchases, 30 days purchases
- price sensitivty

article features
- info 
- trend, 7 days, 30 days purchase for this product group
- textual embedding from category 
- price

user x article features
- did user already buy this article
- age article preferences
- cross id features (ID Matching)

contextual features
- day (weekend, weekday)

sparse features
- user id
- article id
- articles bought by the users in last 7 days
- dense bucketization for user features
- dense bucketization for item features

embedding features
- article embedding from the picture, image understanding model 
- create CNN to predict product group and color and than use those embeddings for clustering, use the cluster_id as a feature
- user x cluster_id purchased in the past information

# Modelling

two stage ranking 

retrieval 

injection of post popular articles in inventory

TTSN
- user tower
- item tower
- KNN on embeddings
- different tower size based on  training data size
- at the top either cross product with sigmoid, or concat and MLP

ranking

GBDT ?
factorizaion machines
deep and wide
SparseNN

Factorization Machine to capture all the feature-feature interaction
Deep NN to capture non linearities

**model size**
- think about how many parameters this model should be, how many GPUs we should use

# Offline Evaluation

