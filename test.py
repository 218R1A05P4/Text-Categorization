import os
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from apyori import apriori
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import accuracy_score
from numpy import dot
from numpy.linalg import norm
import math
from collections import Counter

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Path to the dataset
path = 'dataset'

labels = []
document = []
Y = []

# Function to get the index of a label
def getID(name):
    if name in labels:
        return labels.index(name)
    else:
        return -1

# Walk through directories and gather labels
for root, dirs, files in os.walk(path):
    for name in dirs:
        if name not in labels:
            labels.append(name)
print("Labels:", labels)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess documents
def cleanDoc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Process each file
for root, dirs, files in os.walk(path):
    for filename in files:
        doc = ''
        name = os.path.basename(root)
        with open(os.path.join(root, filename), "r", encoding='utf-8') as file:
            for line in file:
                line = line.strip().lower()
                doc += line + " "
        doc = doc.strip()
        clean = cleanDoc(doc)
        label_id = getID(name)
        if label_id != -1:
            Y.append(label_id)
            document.append(clean)

# Function to get frequent item counts for association rules
def getCount(data):
    count = Counter(data)
    return [item for item, freq in count.items() if freq > 1]

# Association rule mining
if document:
    association_rules = apriori(getCount(document[0].split(" ")), min_support=0.2, min_confidence=0.2)
    association_results = list(association_rules)
    listRules = [list(association_results[i][0]) for i in range(len(association_results))]
    if listRules:
        listRules = listRules[-1]
    else:
        listRules = []
    print("Association Rules:", listRules)

# Vectorize the documents using TF-IDF
stopwords = nltk.corpus.stopwords.words("english")
vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=40)
tfidf = vectorizer.fit_transform(document).toarray()
df = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names_out())
df = df.values
X = df[:, 0:df.shape[1]]
Y = np.asarray(Y)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
print("Feature Matrix X:", X)
print("Labels Y:", Y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Train and evaluate Bernoulli Naive Bayes
nb_cls = BernoulliNB(binarize=0.0)
nb_cls.fit(X_train, y_train)
nb_predictions = nb_cls.predict(X_test)
nb_acc = accuracy_score(y_test, nb_predictions) * 100
print("BernoulliNB Accuracy:", nb_acc)

# Train and evaluate Decision Tree Classifier
dt_cls = DecisionTreeClassifier()
dt_cls.fit(X_train, y_train)
dt_predictions = dt_cls.predict(X_test)
dt_acc = accuracy_score(y_test, dt_predictions) * 100
print("DecisionTreeClassifier Accuracy:", dt_acc)

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    return float(numerator) / denominator if denominator else 0.0

# Function to print feature names based on non-zero entries
def getColumns(a, b):
    feature_names = vectorizer.get_feature_names_out()
    for i in range(len(a)):
        if a[i] > 0 and b[i] > 0:
            print(feature_names[i])

# Evaluate using cosine similarity
similarity_acc = 0
for i in range(len(X_test)):
    pval = 0
    nval = 0
    for j in range(len(X_train)):
        predict_score = dot(X_train[j], X_test[i]) / (norm(X_train[j]) * norm(X_test[i]))
        if predict_score > 0:
            pval += 1
        else:
            nval += 1

    if pval > 0:
        pval /= X_train.shape[0]
        getColumns(X_train[j], X_test[i])
        similarity_acc = pval
print("Similarity-Based Accuracy:", similarity_acc)
