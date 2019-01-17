import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer

## loading the stackoverflow dataset into dataframe
df = pd.read_json("https://raw.githubusercontent.com/kavgan/nlp-text-mining-working-examples/master/tf-idf/data/stackoverflow-data-idf.json", lines=True)

## function to lowercase all the words, remove tags, digits and special characters from the dataset..
def cleaning(data):
    data = data.lower()

    data = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", data)

    data = re.sub("(\\d|\\W)+", " ", data)

    return data

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

## concatinating the title and the body of the content of the data
df['data'] = df['title'] + df['body']
df['data'] = df['data'].apply(lambda x: cleaning(x))

# load a set of stop words of nltk or can be loaded own defined stopwords here..
stopwords = set(stopwords.words('english'))

# get the data column
docs = df['data'].tolist()

# create a vocabulary of words,
# ignore words that appear in 85% of documents,
# eliminate stop words
cv = CountVectorizer(max_df=0.85, stop_words=stopwords, max_features=10000)
word_count_vector = cv.fit_transform(docs)

tfidf_transformer=TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

## loading the testing dataset into dataframes
df_test = pd.read_json("https://raw.githubusercontent.com/kavgan/nlp-text-mining-working-examples/master/tf-idf/data/stackoverflow-test.json", lines=True)
df_test['text'] = df_test['title'] + df_test['body']
df_test['text'] = df_test['text'].apply(lambda x: cleaning(x))

# get test docs into a list
docs_test = df_test['text'].tolist()

# you only needs to do this once, this is a mapping of index to
feature_names = cv.get_feature_names()

# get the document that we want to extract keywords from
doc = docs_test[0]

# generate tf-idf for the given document
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

# sort the tf-idf vectors by descending order of scores
sorted_items = sort_coo(tf_idf_vector.tocoo())

# extract only the top n; n here is 10
keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

# now print the results
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k, keywords[k])