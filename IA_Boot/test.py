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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

discours = Bunch()
discours.filenames = []
discours.data = []

# Folder Path
path = "/home/tom/Documents/Cours/IA/tor_2021_24/discours/ex"
  
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


my_dict = {}
keys = []
escapes = ''.join([chr(char) for char in range(1, 32)])
translator = str.maketrans('', '', escapes)
for i in discours["filenames"]:
    keys.append(i)

data = []
for i in discours["data"]:
    data.append(i)


#populate my dict with filenames and data with escaping \n
for i in range(len(keys)):
    my_dict[keys[i]] = data[i][0:40]#.translate(translator)




#df = pd.DataFrame.from_dict(my_dict, orient="index")


def create_bag_of_words(data):

    #tokenize with lower + only alphanumerique caract
    tokens = [w for w in word_tokenize(data.lower()) if w.isalpha()] 

    #Remove franch stop words
    no_stops = [t for t in tokens if t not in stopwords.words('french')]

    #Lemmatize my data
    wordnet_lemmatizer = WordNetLemmatizer()

    bow_lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
    return bow_lemmatized

bow_lemmatized_speeches = []
for speech in discours["data"]:
    speech = speech.replace("a-","a ")
    bow_lemmatized_speeches.append(create_bag_of_words(speech))


dictionary = Dictionary(bow_lemmatized_speeches)

corpus = [dictionary.doc2bow(speech) for speech in bow_lemmatized_speeches]

bow_doc = sorted(corpus[0], key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    print(word_id)
    print(dictionary.get(word_id), word_count)

total_word_count = defaultdict(int)

for word_id, word_count in itertools.chain.from_iterable(corpus):
    
    total_word_count[word_id] += word_count


tfidf = TfidfModel(corpus)

tfidf_weights = tfidf[corpus[0]]
tfidf_weights1 = tfidf[corpus[1]]


df = pd.DataFrame(tfidf_weights)
df1 = pd.DataFrame(tfidf_weights1)


print(df.shape)
print(df.head())

