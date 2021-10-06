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
path = "S:\Dev\DataAnalysis\Data\IA_Boot\discours\\tous"
  
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

df = pd.read_csv("Alain.csv")


a = 0

wordnet_lemmatizer = WordNetLemmatizer()

for i in range(len(discours["filenames"])):
    df.loc[a] = [discours["filenames"][a], wordnet_lemmatizer.lemmatize(discours["data"][a].lower())]
    a = a+1
print(df)


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
    values.append(string)
    bow_lemmatized = []
    string = ""


vectorizer = TfidfVectorizer(stop_words=final_stopwords_list)
features = vectorizer.fit_transform(values)
k = 2
model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1)
model.fit(features)
df["cluster"] = model.labels_
print("Cluster centroids: \n")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(k):
    print("Cluster %d:" % i)
    for j in order_centroids[i, :10]: #print out 10 feature terms of each cluster
        print (' %s' % terms[j])
    print('------------')

