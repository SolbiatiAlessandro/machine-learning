
from typing import Reversible


class DB:
    def __init__(self):
        self.tables = {}

        pass

    def insert(self, table: str, data: dict):
        # db.insert("users", {"id": "1", "name": "Ada", "birthday": "1815-12-10"})
        # adding one row per insert
        if table not in self.tables.keys():
            self.tables[table] = {} # create table user
        row_id = data["id"] # 1 
        self.tables[table][row_id] = data # 
        

    def query(self, table, column: list[str], where=None, order_by=[]) -> list[str]:
        # where_column, where_value, where_operator
        # return all the rows for that column

        # where: (bday, "2024-10-01", "="), (name, "adam", ">")

        table = self.tables[table]
        res = []
        for row_id, row_value in table.items():
            # 1, {"id": "1", "name": "Ada", "birthday": "1815-12-10"}
            # row_value -> dictionary

            # not really efficient
            #import pdb;pdb.set_trace()
            row_result = {}
            for column_id, column_value in row_value.items():

                operator_map = {
                    "=": lambda x, y: x==y,
                    ">": lambda x, y: x>y,
                    "<": lambda x, y: x<y,
                }
                # iterating over rows
                where_conditition_met = True
                if where:
                    for where_column, where_value, where_operator in where:
                        
                        # age where_value: 10

                        existing_value = row_value[where_column] 
                        comparison_function = operator_map[where_operator]
                        if comparison_function(existing_value, where_value) is not True:
                            where_conditition_met = False

                if where_conditition_met:
                    if column_id in column:
                        row_result[column_id] = column_value
            
            if len(row_result.keys()) > 0:
                res.append(row_result)

        # {"name": "Ada", "birthday": "1815-12-10"}, {"name": "Charles", "birthday": "1791-12-26"}
        if not order_by:
            return res
        # order_by = order_by[0] # TOOD: change for more columns
        # order_by is in the where

        def lt_get_comparison_function(x, y, order_by):
            for order in order_by:
                if x.get(order) != y.get(order):
                    return x.get(order) < y.get(order)

        def sort_function(row_value):
            # "age", 3
            res = []
            for element in order_by: # "name","age"]
                #multiplier = 1
                #if column_descending:
                #    multiplier = -1 
                res.append(row_value.get(element))
            return res
    
        ordered_res = sorted(res, key=sort_function) #[::-1]
        return ordered_res 


# initi

def test():
    db = DB()
# 
    db.insert("users", {"id": "1", "name": "Ada", "birthday": "1815-12-10"})
    db.insert("users", {"id": "2", "name": "Charles", "birthday": "1791-12-26"})
    result = db.query("users", ["name"])
    print(result)
    assert result == [{"name": "Ada"}, {"name": "Charles"}]
    print("Test 1")

    result = db.query("users", ["name", "birthday"])
    assert result == [{"name": "Ada", "birthday": "1815-12-10"}, {"name": "Charles", "birthday": "1791-12-26"}]
    print("Test 2")

    result = db.query("users", ["name", "birthday"],
    [("birthday", "1800-01-01", ">")])
    print(result)
    assert result == [{"name": "Ada", "birthday": "1815-12-10"}]
    print("Test 3")


    result = db.query("users", ["name"],
    [("birthday", "1800-01-01", ">")])
    print(result)
    assert result == [{"name": "Ada"}]
    print("Test 4")

    result = db.query("users", ["name"],
    [("birthday", "1815-12-10", "=")])
    print(result)
    assert result == [{"name": "Ada"}]
    print("Test 5")

    result = db.query("users", ["name"],
    [("birthday", "1800-12-10", "<")])
    print(result)
    assert result == [{"name": "Charles"}]
    print("Test 6")

    result = db.query("users", ["name"],
    [("birthday", "1800-12-10", "<"), ("name", "Charles", "=")])
    print(result)
    assert result == [{"name": "Charles"}]
    print("Test 7")

    result = db.query("users", ["name"],
    [("birthday", "1800-12-10", "<"), ("name", "Charless", "=")])
    print(result)
    assert result == []
    print("Test 8")

    # order vy
    db = DB()
# 
    db.insert("users", {"id": "1", "name": "Zda", "birthday": "1815-12-10"})
    db.insert("users", {"id": "2", "name": "Charles", "birthday": "1791-12-26"})
    db.insert("users", {"id": "3", "name": "Ada", "birthday": "1791-12-26"})
    result = db.query("users", ["name"], order_by=["name"])
    print(result)
    assert result == [{"name": "Ada"},{"name": "Charles"},{"name": "Zda"}]
    print("Test 9")

    db.insert("users", {"id": "1", "name": "Zda", "birthday": "1815-12-10"})
    db.insert("users", {"id": "2", "name": "Charles", "birthday": "1791-12-26"})
    db.insert("users", {"id": "3", "name": "Zda", "birthday": "1791-12-26"})
    result = db.query("users", ["name", "birthday"], order_by=["name", "birthday"])
    print(result)
    assert result == [{"name": "Charles", "birthday": "1791-12-26"},{"name": "Zda", "birthday": "1791-12-26"},{"name": "Zda", "birthday": "1815-12-10"}]
    print("Test 10")
    


    # where one column equal one value
    # where one column great
    # where one column smaller
    # where one column great than all -> empty result
    # where one column smaller than all -> empty result

    # query a column but we where another column that we are not query

    # query(name), where(birhtday)


test()
    

# O(N)
# go thoruhg all the results

# O(logN)
# sorted index lists
# b1 <b2> b3
# 10^6

    #db.insert("users", {})
    # db.insert("users", {"id": "1"})


# should return [{"name": "Ada",}, {"name": "Charles"}]



# memoery
# O(number rows * number of columns)
# time
# O(number rows)

# for developers
# no perforamnce
# good API
# reasonable test coverage
# in memory 

# start with some tests 
# TDD


We would like to make ChatGPT more useful for knowledge work by giving the model access to private corpora. Assume that organizations can upload their documents to OpenAI. These documents form the organization’s knowledge repository (e.g. product roadmaps, meeting notes, onboarding guides, etc.) and can be in text, pdf, doc and ppt format.

For this question, assume that the corpus of documents is static, fully made available to OpenAI and that we do not need to worry about permissions. To protect sensitive information we cannot train the LLM used by ChatGPT directly on these documents.

Assume access to compute, limited human annotation resources and open source or proprietary models (but be prepared to discuss how they work).

In this interview, **we want to build an extension of ChatGPT that improves its ability to answer user questions based on these documents**. By extension we mean that ChatGPT will continue to be able to assist the user with all the typical requests it currently does, but in addition to those it **will rely on information from the corporate documents when the user asks a question related to their work.**

The goal for this interview is to first arrive at an overall system design (with a focus on ML techniques) for how this extension will work. We will then drill down into some of the components proposed to understand how they work in detail.


1. high level design
2. then we can go deeper
3. 1.25

we want to build an extension of ChatGPT that improves its ability to answer user questions based on these document

RAG retriveal augmented generation
retreival system

A               B                     C            D
<input data> -> <retrieval system> -> <ChatGPT> -> <User>


# Requirements

A
- corpus of documents is static
- big, 100M documents
- text, pdf, doc and ppt format
- product roadmaps, meeting notes, onboarding guides,
B
- latency, not worry ~  100ms
C
- latency, not worry ~ 1sec
- "corporate search"
- general ("what is the product X") and/or specific ("how do we use X to do Y")
D
- factuality, agent is based on the documnet
- metrics: user satisfaction, click, time spent on the source

# Simple System Design
A
- 100M document
- list of documents
B
- retrieval system : give most releveant documents
- 100M -> 100/20
- multi stage
- A1 100M -> 1000/10000 -> A2 100
- A1 (model)
- <document> -> [0.1, 0.2]
- neighbour search, query -> [0,1, 0.2]
- A2
- rank your document 1000 -> 100
- model moth <document and the query>, heavier model
- relevance, precision, and recall
C
- chat uses 100 document in the reponse
- inject this in the prompt of the LLM
- answer only with this documents {contenxt}
D

# deeper

A1
100M

splits chunks
chunk <0, 500>
every have document, metadata
title, link, how many clicks
product roadmaps, meeting notes, onboarding guides,
document contain images images


100M
index them
vector datbases

hyeriarchy
100M -> 10M -> 1M




#images
screenshots, images always text
OCR - from image back to text -> give back to text pipelines

embedding from the images

roadmp is power points
long doucment
notes not well formmated

chunk <0, 500>  + metadata append

B1 - B2

encoders: take text give you encoding
encode only the document

e1 e2 the represtaion must be comparable
loss we want to minise e1 - e2 (cosine distance)

pretrained model
domina specific

fine tuning model on domain specific data
label - do we have lables?

BERT(d1) = embeddings
contrastive learning, [d1, d1] = 1 , [d1, d3] = 0

loss = cosine dinstance(BERT(d1), BERT(d2))
labels

unsuprvised
masking

BERT(d1 maksed 1)
BERT(d1 masked 2)
[d1 masked 1, d1 masked 2 ] = 1

high recall BERT

baselines
word frequency
lexical search

hybrid approach
what is product X
"how do we use X for Y"
reformualte the query with LLM

"how do we use X for Y" -> what are some good retreival queries
"what is X"
"wahts is Y"


metric we are optimzing recall

BERT
Bidirectional Encoder Representation Transformer
Transformer architeur
finde represenation
encoding

what's is the input to encode


A2 O(# documents * # queryies)

cross-encoders
you encode both document and query
100 * 100M

ranking part

1000 docment + 1 query
100

model(document, query) = simialirty
model(document, query) = relevenat

nDCG ~ gain = relevance
nDCG@k = for each k : gain(k) / k

cross-encoder, encodes both
model -> BERT
point wise ranking -> score
list wise ranking

LLM at the re-rank them

incorpoate features?
type of document, length of document, how much the doc was used in last 30 days
type of document = 10, 25
Sparse features

labels - generator is

tokens -> embedding
DNN, attention
deep capture non lineary
wide feature interaction

C LLM:
10

query_user

prompt injection
context window of the LLM
millions of tokens

"""
user:{query} use ONLY THIS and not else this results to answer query {ranking output}
"""

does llm use the document or not?
<a> <b> <c> -> nhow many of the sentences are supported by a document?
context recall

"did trump win the election"
<a> <b>

facautlity evaluation
labelling

LLM-as-a-judget

collect feedback

thumbs, thumbs down, report
positive: jhow mcuch did user spend in th

post-training
RLHF reinformcnet learning from human feedback
domain specific
"in the heart there .. jaragon"

user input
user quality

rule base check - who is using the sytem (employee depearment)
LLM check
privacy: "how much mony does my boss make" -> filter the document, at indexing stage, after the retrieval
safety: "how do I downlaod the company data to my USB key" -> make the model aligned ont he prefere of the company
RLHF

document has privacy check, viewer context who is viewing the contenxt

good testing
red teaming

quality suggestions
autocomplete
but don't have much data

"how does project X .. "
> "realted to Y"
> "used for login

caching older results, what's trending
100 users , sort how trendy query is
cache the retreival

personalize to the user
what is teh deperatment of the user

cold start
users wont' know

we can look at the inventory first, what kind fo dcument
most of the document , how the heart works
clustering on the documents
documents in the cluster number

pre-compute

smaller LLM to do the autocomplete

outside of the corpus "what's the holiday this year"

Agentic LLM
do I need to use the retriveal system or not
LLM acces to the retrieval

force the LLM to invoceate

get query and get poor results, estimated relevance low

"what's the holiday this year"
take the emnbedding
1) and then we check the distnace

e1 - e2 = 10

2) caching, before did we use the corpus or not for this quetsion

