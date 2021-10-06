# Import Module
import os
from bunch import Bunch
import pandas as pd
import itertools  
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
from gensim.models.tfidfmodel import TfidfModel
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from nltk.corpus import stopwords

final_stopwords_list = stopwords.words('french')
final_stopwords_list.extend(["cette","si", "tout","mais","cela", "bien","même","parce","ceux","plus","ãªtre","mãªme","aussi","sans","temp","comme","tous","pay","faut","fait","an","quand","alors","faire","veux","class","non","oui","rien","leurs","vie","lã","va","toutes","dire","dit","peut","encore","entre","depuis","doit","jamais","dont","deux","voilã","donc","moins","sarkozy","chirac","grand","corse","gauche","droite","car","peu","jospin","hollande","premier","madelin","marseille","merci","remercie"])
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

discours = Bunch()
discours.filenames = []
discours.data = []

# Folder Path
path = "S:\Dev\DataAnalysis\Data\IA_Boot\discours\\all"
  
# Change the directory
os.chdir(path)
  

#Create my bunch object  
def create_bunch(file_path):
    with open(file_path, 'r') as f:
        discours.data.append(f.read())
  
  
# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"
        #Remove weird caract from file title
        file = file.replace("\xE7","c")
        file = file.replace("\xE9","e")
        file = file.replace("\xE8","e")

        discours.filenames.append(file)

        create_bunch(file_path)

df = pd.read_csv("Discours.csv")
print(df.columns)
a = 0

wordnet_lemmatizer = WordNetLemmatizer()

for i in range(len(discours["filenames"])):
    if (discours["filenames"][a].split("_")[1] == "Madelin" or discours["filenames"][a].split("_")[1] == "Megret" or discours["filenames"][a].split("_")[1] == "Lepage" or discours["filenames"][a].split("_")[1] == "Bayrou" or discours["filenames"][a].split("_")[1] == "Chirac" or discours["filenames"][a].split("_")[1] == "Le" or discours["filenames"][a].split("_")[1] == "Chevènement" or discours["filenames"][a].split("_")[1] == "Sarkozy" or discours["filenames"][a].split("_")[1] == "Hue"):
        df.loc[a] = [discours["filenames"][a].split("_")[1], wordnet_lemmatizer.lemmatize(discours["data"][a].lower()),"1","0"]
    else:
        df.loc[a] = [discours["filenames"][a].split("_")[1], wordnet_lemmatizer.lemmatize(discours["data"][a].lower()),"0","1"]
    a = a+1

bow_lemmatized = []
string = ""
values = []
for content in df["content"]:
    tokens = [w for w in word_tokenize(content.lower()) if w.isalpha()]

    #Remove franch stop words
    no_stops = [t for t in tokens if t not in stopwords.words('french')]
    #Lemmatize my data
    bow_lemmatized += [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
    for t in bow_lemmatized:
        string += " " + t
    #values.append(string)
    content = string
    bow_lemmatized = []
    string = ""


y = df["droite"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df["content"], y, test_size=0.33, random_state=53)
print(type(Ytrain))
count_vectorizer = CountVectorizer(stop_words=final_stopwords_list)
count_train = count_vectorizer.fit_transform(Xtrain)
count_test = count_vectorizer.transform(Xtest)


tfidf_vectorizer = TfidfVectorizer(stop_words=final_stopwords_list, max_df=0.7)
# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(Xtrain)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(Xtest)

nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, Ytrain)
pred = nb_classifier.predict(count_test)
score = metrics.accuracy_score(Ytest, pred)
print(score)
cm = metrics.confusion_matrix(Ytest, pred, labels=["0","1"])
print(cm)

nb_classifier2 = MultinomialNB()
nb_classifier2.fit(tfidf_train, Ytrain)
pred2 = nb_classifier2.predict(tfidf_test)
score2 = metrics.accuracy_score(Ytest, pred2)
print(score2)
cm2 = metrics.confusion_matrix(Ytest, pred2, labels=["0","1"])
print(cm2)