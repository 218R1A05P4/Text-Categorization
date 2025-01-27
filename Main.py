from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from apyori import apriori
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
import os


main = tkinter.Tk()
main.title("Integrated Text Categorization System")
main.geometry("1300x1200")

global filename
document = []
Y = []
global nb_acc, dt_acc, propose_acc
labels = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
path = 'dataset'
def readLabel():
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index  

def cleanDoc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens                

def uploadDataset():
    textarea.delete('1.0', END)
    global filename
    global dataset
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)    
    textarea.insert(END,'Dataset loaded\n\n')
       
def naiveBayes():
    global nb_acc
    textarea.delete('1.0', END)
    document.clear()
    Y.clear()
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            doc = ''
            name = os.path.basename(root)
            with open(root+"/"+directory[j], "r") as file:
                for line in file:
                    line = line.strip('\n')
                    line = line.strip()
                    line = line.lower()
                    doc+=line+" "
            file.close();
            doc = doc.strip()
            clean = cleanDoc(doc)
            Y.append(getID(name))
            document.append(clean)
    listRules = document[0]
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=40)
    tfidf = vectorizer.fit_transform(document).toarray()        
    df = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names_out())
    textarea.insert(END,str(df))
    df = df.values
    X = df[:, 0:df.shape[1]]
    Y1 = np.asarray(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y1 = Y1[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.2)
    cls = GaussianNB() #gaussian Naive Bayes
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test) 
    nb_acc = accuracy_score(y_test,prediction_data)*100
    textarea.insert(END,"\nNaive Bayes Accuracy : "+str(nb_acc)+"\n")
    textarea.insert(END,"Naive Bayes Train Document Size : "+str(len(listRules) * df.shape[0])+"\n")
    
def decisionTree():
    global dt_acc
    listRules = document[0]
    textarea.delete('1.0', END)
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=40)
    tfidf = vectorizer.fit_transform(document).toarray()        
    df = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names_out())
    textarea.insert(END,str(df))
    df = df.values
    X = df[:, 0:df.shape[1]]
    Y1 = np.asarray(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y1 = Y1[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.2)
    cls = DecisionTreeClassifier() #decision tree classifier
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test) 
    dt_acc = accuracy_score(y_test,prediction_data)*100
    textarea.insert(END,"\nDecision Tree Accuracy : "+str(dt_acc)+"\n")
    textarea.insert(END,"Decision Tree Train Document Size : "+str(len(listRules) * df.shape[0])+"\n")

def getColumns(a,b,names):
    name = []
    for i in range(len(a)):
        if a[i] > 0 and b[i] > 0:
            if names[i] not in name:
                name.append(names[i])
    return name            

def getCount(data):
    temp = []
    for i in range(len(data)):
        value = data[i]
        count = data.count(value)
        if count > 1:
            temp.append(value)
    print(temp)
    return [temp]

propose_acc = 0
nb_acc = 0
dt_acc = 0
propose_acc = 0

def proposeAssociationRule():
    global propose_acc
    textarea.delete('1.0', tkinter.END)
    #calculating association rules
    association_rules = apriori(getCount(document[0].split(" ")), min_support=0.2, min_confidence=0.2)
    association_results = list(association_rules)
    print(association_results)
    listRules = [list(association_results[i][0]) for i in range(0,len(association_results))]
    listRules = listRules[len(listRules) -1]
    textarea.insert(END,"Association Rules Frequent Pattern : "+str(listRules)+"\n")

    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    #calculating features
    vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=40)
    tfidf = vectorizer.fit_transform(document).toarray()        
    df = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names_out())
    textarea.insert(END,str(df))
    df = df.values
    X = df[:, 0:df.shape[1]]
    Y1 = np.asarray(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y1 = Y1[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    names = vectorizer.get_feature_names_out()
    features = []
    for i in range(len(X_test)):
        pval = 0
        nval = 0
        for j in range(len(X_train)):
            predict_score = dot(X_train[j], X_test[i])/(norm(X_train[j])*norm(X_test[i]))
            if predict_score > 0:
                pval = pval + 1 #propose calculating pval
            else:
                nval = nval + 1 #propose calculating nval

        if pval > 0:
            pval = pval / X_train.shape[0] #getting percentage of positive documents
            col_names = getColumns(X_train[j],X_test[i],names)
            for k in range(len(col_names)):
                if col_names[k] not in features:
                    features.append(col_names[k])
            propose_acc = pval * 100
    textarea.insert(END,"\nPropose Association Rule Accuracy : "+str(propose_acc)+"\n")
    textarea.insert(END,"Propose Algorithm Train Document Size : "+str(len(listRules) * df.shape[0])+"\n")
    textarea.insert(END,'Test words found in train documents ; '+str(features)+"\n\n\n")
    
def graph():
    height = [nb_acc, dt_acc, propose_acc]
    bars = ('Naive Bayes Accuracy','Decision Tree Accurcay','Propose Accuracy')
    y_pos = np.arange(len(bars))
    colors = ['blue', 'green', 'red']
    plt.bar(y_pos, height, color=colors)
    plt.xticks(y_pos, bars)
    plt.show()
    
def close():
    main.destroy()
    
font = ('times', 15, 'bold')
title = Label(main, text='Integrated Text Categorization System')
title.config(bg='mint cream', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Abstract Text Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='mint cream', fg='olive drab')  
pathlabel.config(font=font1)           
pathlabel.place(x=320,y=100)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=naiveBayes)
nbButton.place(x=50,y=150)
nbButton.config(font=font1) 

dtButton = Button(main, text="Run Decision Tree Algorithm", command=decisionTree)
dtButton.place(x=320,y=150)
dtButton.config(font=font1) 

parButton = Button(main, text="Run Propose Associtaion Rue Based Algorithm", command=proposeAssociationRule)
parButton.place(x=600,y=150)
parButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=200)
graphButton.config(font=font1) 

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=360,y=200)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=20,width=150)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=250)
textarea.config(font=font1)

readLabel()
main.config(bg='gainsboro')
main.mainloop()
