import pandas as pd
def read_data():
    customers = pd.read_csv("./data/customers.csv")
    articles = pd.read_csv("./data/articles.csv")
    transactions = pd.read_csv("./data/transactions_train.csv")

def train_test(factor=250000):
    train = transactions[-5*factor:-factor]
    test = transactions[-factor:]

def get_labels(dataset):
    positive_labels = dataset[['customer_id', 'article_id']]
    positive_labels['label'] = 1.0

    customer_positive = positive_labels.groupby('customer_id')['article_id'].apply(set)

    # Define the helper function that will sample a negative article for a row.
    def sample_negative_for_row(row):
        customer = row['customer_id']
        pos_set = customer_positive.get(customer, set())
        
        # Start by sampling a candidate from the entire articles list.
        candidate = np.random.choice(articles['article_id'])
        
        # Iterate until the candidate is not in the customer's positive set.
        while candidate in pos_set:
            candidate = np.random.choice(articles['article_id'])
        
        # Increment our counter and print progress every 1,000 rows.
        sample_negative_for_row.counter += 1
        if sample_negative_for_row.counter % 1000 == 0:
            print(f"Processed {sample_negative_for_row.counter} rows")
        
        return candidate
    
    # Initialize a counter attribute on the function.
    sample_negative_for_row.counter = 0
    
    print("\nStep 2: Generating negative samples for each positive interaction using .apply...")
    # For each row in positive_labels, sample a negative article that is not in the customer's positive set.
    positive_labels['negative_article'] = positive_labels.apply(sample_negative_for_row, axis=1)

    # Step 1: Create a new DataFrame for negative samples
    negative_labels = positive_labels[['customer_id', 'negative_article']].copy()
    negative_labels.rename(columns={'negative_article': 'article_id'}, inplace=True)
    negative_labels['label'] = 0.0
    
    # Step 2: Concatenate positive and negative labels into one DataFrame
    full_labels = pd.concat([positive_labels[['customer_id', 'article_id', 'label']], negative_labels], ignore_index=True)

    return full_labels

    
    
